# DeCo-Diff-Gating

**基于扩散模型的多类别无监督异常检测**

本仓库是 CVPR 2025 论文 [**"Correcting Deviations from Normality: A Reformulated Diffusion Model for Unsupervised Anomaly Detection"**](https://openaccess.thecvf.com/content/CVPR2025/papers/Beizaee_Correcting_Deviations_from_Normality_A_Reformulated_Diffusion_Model_for_Multi-Class_CVPR_2025_paper.pdf) 的 PyTorch 实现。

本分支额外添加了 **门控机制 (DoD-Gating & Skip-Gating)**、**多卡训练/推理支持** 和 **Gradio Web 界面**。

---

## 📂 项目结构

```
DeCo-Diff/
├── train_DeCo_Diff.py          # 分布式训练脚本
├── train_classifier.py          # 类别分类器训练
├── evaluation_DeCo_Diff.py      # 单卡评估
├── evaluation_DeCo_Diff_DDP.py  # 多卡评估
├── inference_single.py          # 单张图片推理
├── inference_auto.py            # 自动类别识别推理
├── app_gradio_auto.py           # Gradio Web 界面
├── models.py                    # 模型工厂
└── ldm/modules/diffusionmodules/
    └── openaimodel.py           # UNet + 门控机制
```

---

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
pip install gradio  # Web 界面
```

### 数据集

- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

---

## 🏋️ 训练

### 单卡训练

```bash
torchrun --nnodes=1 --nproc_per_node=1 train_DeCo_Diff.py \
    --dataset mvtec \
    --data-dir ./mvtec-dataset \
    --object-category all \
    --model-size UNet_L \
    --epochs 800
```

### 多卡训练

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_DeCo_Diff.py \
    --dataset mvtec \
    --data-dir ./mvtec-dataset \
    --object-category all \
    --global-batch-size 256
```

### 断点续训

```bash
torchrun train_DeCo_Diff.py --resume ./checkpoints/last.pt
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec | 数据集 (mvtec/visa) |
| `--object-category` | all | 类别 (all 或特定类别) |
| `--model-size` | UNet_L | 模型大小 (XS/S/M/L/XL) |
| `--epochs` | 800 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--center-size` | 256 | 裁剪尺寸 |
| `--resume` | - | 断点续训路径 |

---

## 🧪 评估

### 单卡评估

```bash
python evaluation_DeCo_Diff.py \
    --dataset mvtec \
    --data-dir ./mvtec-dataset \
    --model-path ./checkpoints/epoch-660.pt \
    --object-category all
```

### 多卡评估

```bash
torchrun --nnodes=1 --nproc_per_node=4 evaluation_DeCo_Diff_DDP.py \
    --dataset mvtec \
    --data-dir ./mvtec-dataset \
    --model-path ./checkpoints/epoch-660.pt \
    --object-category all
```

---

## �️ Web 界面

### 训练分类器（首次使用）

```bash
python train_classifier.py --data-dir ./mvtec-dataset --epochs 30
```

### 启动 Gradio

```bash
python app_gradio_auto.py \
    --model-path ./checkpoints/epoch-660.pt \
    --classifier-path ./classifier_mvtec.pth

# 公网访问
python app_gradio_auto.py \
    --model-path ./checkpoints/epoch-660.pt \
    --classifier-path ./classifier_mvtec.pth \
    --share
```

### 功能特性

- 🔍 **自动类别识别** - 无需手动选择类别
- ⚡ **实时检测** - 上传图片即可检测
- 📊 **可视化结果** - 重建图像、异常热力图、区域标注
- 🔧 **参数可调** - 阈值、门控融合等

---

## 📝 单张推理

```bash
# 基本用法
python inference_single.py \
    --image ./test.jpg \
    --model-path ./checkpoints/epoch-660.pt \
    --class-id 0 \
    --threshold 0.3

# 使用门控融合
python inference_single.py \
    --image ./test.jpg \
    --model-path ./checkpoints/epoch-660.pt \
    --class-id 0 \
    --use-gate-fusion true
```

### 自动类别识别推理

```bash
python inference_auto.py \
    --image ./test.jpg \
    --model-path ./checkpoints/epoch-660.pt \
    --classifier-path ./classifier_mvtec.pth
```

---

## 🔧 门控机制

本分支在原始 DeCo-Diff 基础上添加了门控机制：

- **DoD-Gating**: 在 UNet 输出层添加门控，自适应调节偏差预测强度
- **Skip-Gating**: 对 skip connection 添加门控，控制特征传递

详细文档请参阅 [GATING_README.md](./GATING_README.md)

---

## 📄 许可证

MIT License
