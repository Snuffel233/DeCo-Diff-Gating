# DeCo-Diff 技术文档

## 项目概述

DeCo-Diff (Deviation-Compensated Diffusion) 是一种基于扩散模型的工业视觉异常检测方法。本项目实现了完整的训练、评估和推理流程，并增加了门控机制 (DoD-Gating 和 Skip-Gating) 来提升检测精度。

---

## 目录

1. [系统架构](#1-系统架构)
2. [环境配置](#2-环境配置)
3. [训练流程](#3-训练流程)
4. [推理流程](#4-推理流程)
5. [Web 界面](#5-web-界面)
6. [门控机制](#6-门控机制)
7. [API 参考](#7-api-参考)

---

## 1. 系统架构

### 1.1 核心组件

```
DeCo-Diff/
├── train_DeCo_Diff.py          # 分布式训练脚本
├── train_classifier.py          # 类别分类器训练
├── evaluation_DeCo_Diff.py      # 单卡评估脚本
├── evaluation_DeCo_Diff_DDP.py  # 多卡评估脚本
├── inference_single.py          # 单张推理脚本
├── inference_auto.py            # 自动分类推理脚本
├── app_gradio_auto.py           # Gradio Web 界面
├── models.py                    # 模型工厂函数
├── diffusion/                   # 扩散模型核心
│   └── gaussian_diffusion.py
└── ldm/modules/diffusionmodules/
    └── openaimodel.py           # UNet 模型定义
```

### 1.2 数据流

```
Input Image → VAE Encoder → Latent Space → Diffusion Sampling → VAE Decoder → Reconstructed Image
                                ↓
                        Anomaly Map Calculation
                                ↓
                        Geometric Mean Fusion
                                ↓
                        Anomaly Score & Detection Result
```

---

## 2. 环境配置

### 2.1 依赖安装

```bash
pip install torch torchvision diffusers transformers
pip install scikit-image scipy scikit-learn pandas
pip install anomalib gradio matplotlib pillow
```

### 2.2 数据集准备

```
mvtec-dataset/
├── bottle/
│   ├── train/good/
│   └── test/
│       ├── good/
│       ├── broken_large/
│       └── ...
├── cable/
├── capsule/
└── ...
```

---

## 3. 训练流程

### 3.1 训练命令

```bash
# 单类别训练
torchrun --nnodes=1 --nproc_per_node=1 train_DeCo_Diff.py \
    --dataset mvtec \
    --object-category bottle \
    --epochs 800 \
    --global-batch-size 128 \
    --lr 1e-4

# 全类别训练
torchrun --nnodes=1 --nproc_per_node=4 train_DeCo_Diff.py \
    --dataset mvtec \
    --object-category all \
    --epochs 800 \
    --global-batch-size 256

# 断点续训
torchrun --nnodes=1 --nproc_per_node=1 train_DeCo_Diff.py \
    --dataset mvtec \
    --object-category bottle \
    --resume ./checkpoints/last.pt
```

### 3.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec | 数据集类型 (mvtec/visa) |
| `--object-category` | all | 训练类别 |
| `--epochs` | 800 | 训练轮数 |
| `--global-batch-size` | 128 | 全局批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model-size` | UNet_L | 模型大小 (XS/S/M/L/XL) |
| `--center-size` | 256 | 中心裁剪尺寸 |
| `--image-size` | 288 | 输入图像尺寸 |
| `--mask-ratio` | 0.7 | Mask 比例 |
| `--ckpt-every` | 10 | 保存检查点间隔 |
| `--resume` | '' | 断点续训检查点路径 |

### 3.3 分布式训练

训练使用 PyTorch DistributedDataParallel (DDP)：

```python
# 初始化分布式环境
dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()

# 使用 DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# 每个 epoch 设置 sampler
for epoch in range(epochs):
    sampler.set_epoch(epoch)
```

### 3.4 检查点格式

```python
checkpoint = {
    "model": model.module.state_dict(),
    "opt": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "args": args
}
```

---

## 4. 推理流程

### 4.1 图像预处理

```python
# 1. Resize 到 image_size (288)
img = img.resize((288, 288), Image.BILINEAR)

# 2. Center crop 到 center_size (256)
left = (288 - 256) // 2
top = (288 - 256) // 2
img = img.crop((left, top, left + 256, top + 256))

# 3. 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
```

### 4.2 VAE 编码

```python
# 编码到 latent 空间
encoded = vae.encode(img_tensor).latent_dist.mean.mul_(0.18215)
```

### 4.3 Diffusion 采样

```python
diffusion = create_diffusion(
    'ddim5',                    # DDIM 5 步采样
    predict_deviation=True,     # 预测偏差模式
    sigma_small=False,
    predict_xstart=False,
    diffusion_steps=10
)

latent_samples = diffusion.ddim_deviation_sample_loop(
    model, encoded.shape,
    noise=encoded,
    clip_denoised=False,
    start_t=5,                  # 从第 5 步开始
    model_kwargs=model_kwargs,
    eta=0
)
```

### 4.4 异常图计算

```python
# 图像差异 (max over channels)
image_diff = torch.abs(image_samples - x0).float().mean(axis=0)
image_diff = image_diff.numpy().transpose(1,2,0).max(axis=2)
image_diff = np.clip(image_diff, 0.0, 0.4) * 2.5
image_diff = gaussian_filter(image_diff, sigma=3)

# Latent 差异 (mean over channels)
latent_diff = torch.abs(latent_samples - encoded).float().mean(axis=0)
latent_diff = latent_diff.numpy().transpose(1,2,0).mean(axis=2)
latent_diff = np.clip(latent_diff, 0.0, 0.2) * 5
latent_diff = gaussian_filter(latent_diff, sigma=1)
latent_diff = resize(latent_diff, (256, 256))

# 几何均值融合
anomaly_map = np.sqrt(image_diff * latent_diff)

# 异常分数
anomaly_score = anomaly_map.max()
is_anomalous = anomaly_score > threshold
```

---

## 5. Web 界面

### 5.1 启动命令

```bash
# 本地访问
python app_gradio_auto.py \
    --model-path ./checkpoints/epoch-660.pt \
    --classifier-path ./classifier_mvtec.pth

# 公网访问
python app_gradio_auto.py \
    --model-path ./checkpoints/epoch-660.pt \
    --classifier-path ./classifier_mvtec.pth \
    --share
```

### 5.2 界面功能

| 功能 | 说明 |
|------|------|
| 自动类别识别 | 使用预训练分类器自动识别图像类别 |
| 参数配置 | 可调整阈值、模型大小、反向步数等 |
| 结果可视化 | 展示分类结果、重建图像、异常热力图 |
| 门控融合 | 可选启用 DoD-Gating 融合 |

### 5.3 分类器训练

```bash
python train_classifier.py \
    --data-dir ./mvtec-dataset \
    --epochs 30 \
    --batch-size 32
```

---

## 6. 门控机制

### 6.1 DoD-Gating

在 UNet 输出层添加门控头：

```python
# gate_head 定义
self.gate_head = nn.Sequential(
    normalization(ch),
    nn.SiLU(),
    conv_nd(dims, model_channels, out_channels, 3, padding=1),
    nn.Sigmoid()
)

# forward 中应用
dod_raw = self.out(h)           # 原始 DoD 预测
dod_gate = self.gate_head(h)    # 门控值 g ∈ [0, 1]
dod_output = dod_gate * dod_raw # 门控后输出
```

### 6.2 Skip-Gating

对每级 skip connection 添加门控：

```python
class SkipGate(nn.Module):
    def __init__(self, channels, dims=2):
        super().__init__()
        self.gate_conv = conv_nd(dims, channels, channels, 1)
    
    def forward(self, skip_feat):
        alpha = torch.sigmoid(self.gate_conv(skip_feat))
        return (1 - alpha) * skip_feat, alpha
```

### 6.3 门控融合

```python
# 提取门控图
output = model(encoded, t, return_gate=True, **model_kwargs)
_, dod_gate, skip_gates = output

# 处理门控图
gate_map = dod_gate[0].mean(dim=0).cpu().numpy()
gate_map = resize(gate_map, (256, 256))
gate_map = gaussian_filter(gate_map, sigma=2)

# 融合
anomaly_fused = 0.6 * anomaly_geometric + 0.4 * gate_map
```

---

## 7. API 参考

### 7.1 模型创建

```python
from models import UNET_models

model = UNET_models['UNet_L'](latent_size=32, ncls=15)
```

### 7.2 Diffusion 创建

```python
from diffusion import create_diffusion

diffusion = create_diffusion(
    timestep_respacing='ddim5',
    predict_deviation=True,
    sigma_small=False,
    predict_xstart=False,
    diffusion_steps=10
)
```

### 7.3 类别映射

```python
# MVTec 类别 (按训练顺序)
MVTEC_CLASS_MAP = {
    "capsule": 0, "bottle": 1, "grid": 2, "leather": 3, "metal_nut": 4,
    "tile": 5, "transistor": 6, "zipper": 7, "cable": 8, "carpet": 9,
    "hazelnut": 10, "pill": 11, "screw": 12, "toothbrush": 13, "wood": 14,
}
```

---

## 附录 A: 常见问题

### Q1: 误报率高怎么办？
- 调高阈值 (0.35 → 0.45)
- 确保使用正确的类别 ID
- 检查预处理是否一致

### Q2: 如何断点续训？
```bash
torchrun train_DeCo_Diff.py --resume ./checkpoints/last.pt
```

### Q3: 多卡训练如何设置？
```bash
torchrun --nnodes=1 --nproc_per_node=4 train_DeCo_Diff.py
```

---

## 附录 B: 性能指标

| 数据集 | 模型 | Image AUROC | Pixel AUROC |
|--------|------|-------------|-------------|
| MVTec | UNet_L (all) | ~97% | ~96% |
| ViSA | UNet_L (all) | ~95% | ~94% |

---

*文档版本: 1.0 | 更新日期: 2024-12-30*
