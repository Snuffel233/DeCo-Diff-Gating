# DeCo-Diff 网络结构改进说明文档

## 概述

本次修改为 DeCo-Diff 项目添加了两个网络结构改进：**DoD-Gating（门控头）** 和 **Skip-Gating（跳连通道门控）**。这些改进旨在让网络更好地区分正常区域和异常区域，从而减少假阳性热力图噪点，提升异常检测性能。

---

## 改动 1：DoD-Gating（门控头）

### 动机

DeCo-Diff 的核心思想是"正常 patch 的 DoD=0，异常 patch 的 DoD≠0"。但普通 UNet 在输出 DoD 时，边界和纹理区域容易产生"小幅 DoD 漂移"，导致假阳性。

### 实现原理

在 UNet 最后的特征层分两个头：

1. **原来的 DoD_head**：输出 ε̂（DoD 预测）
2. **新增 gate_head**：输出 `g = sigmoid(conv1x1(feat))`，值域 [0, 1]

最终输出改为：**ε̂_final = g × ε̂**

### 代码位置

**文件**: `ldm/modules/diffusionmodules/openaimodel.py`

**主要修改**：

```python
# 在 UNetModel.__init__ 中添加
self.gate_head = nn.Sequential(
    normalization(ch),
    nn.SiLU(),
    conv_nd(dims, model_channels, out_channels, 3, padding=1),
    nn.Sigmoid()
)

# 在 UNetModel.forward 中修改
dod_raw = self.out(h)  # 原始 DoD 预测
if self.use_gating:
    dod_gate = self.gate_head(h)  # 门控值 g ∈ [0, 1]
    dod_output = dod_gate * dod_raw  # 门控后的输出
```

### 额外收益

推理时 `g`（门控图）本身就是一张"疑似异常区域图"，可以与论文的几何均值异常图做融合，往往会让定位更干净。

---

## 改动 2：Skip-Gating（跳连通道门控）

### 动机

UNet 的 skip connection 会把高频细节直接"抄回去"。一旦 encoder 特征里对异常有响应，就容易把异常也带回 decoder，影响最终异常图的对比质量。

### 实现原理

对每一级 skip connection 的 feature `s` 加一个轻量门控：

```
α = sigmoid(conv1x1(s))
s' = (1 - α) × s
```

其中 `α` 表示"需要改动的程度"：
- 正常区域：α → 0，保留原始特征
- 异常区域：α → 1，抑制原始特征

### 代码位置

**文件**: `ldm/modules/diffusionmodules/openaimodel.py`

**新增 SkipGate 类**：

```python
class SkipGate(nn.Module):
    """轻量级跳跃连接门控模块"""
    def __init__(self, channels, dims=2):
        super().__init__()
        self.gate_conv = conv_nd(dims, channels, channels, 1)
    
    def forward(self, skip_feat, dod_gate=None):
        alpha = th.sigmoid(self.gate_conv(skip_feat))
        if dod_gate is not None:
            # 如果有 DoD gate，将其影响注入
            alpha = alpha * dod_gate
        gated_feat = (1 - alpha) * skip_feat
        return gated_feat, alpha
```

**在 UNetModel 中添加**：

```python
# 创建跳跃门控模块列表
self.skip_gates = nn.ModuleList([
    SkipGate(ch_skip, dims) for ch_skip in self._skip_gate_chans
])
```

---

## 评估代码修改

### 门控图融合

**文件**: `evaluation_DeCo_Diff.py`

新增功能：
1. `extract_gate_maps()` 函数：从模型中提取门控图
2. 修改 `calculate_anomaly_maps()` 函数：支持门控图融合
3. 新增 `--use-gate-fusion` 命令行参数

### 使用方法

```bash
# 常规评估（不使用门控融合）
python evaluation_DeCo_Diff.py --dataset mvtec --object-category bottle

# 使用门控融合评估（推荐用于带 Gating 训练的模型）
python evaluation_DeCo_Diff.py --dataset mvtec --object-category bottle --use-gate-fusion true
```

当启用门控融合时，输出会包含额外的异常图：
- `gate_map`: 门控图本身
- `gate_fused`: 门控图与几何均值异常图的加权融合（0.6 × geometric + 0.4 × gate_map）

---

## 控制开关

### 运行时控制

模型提供 `use_gating` 属性，可以在推理时开关门控功能：

```python
model.use_gating = True   # 启用门控（默认）
model.use_gating = False  # 关闭门控（用于对比实验）
```

### Forward 函数新参数

```python
# 获取门控图（用于可视化或融合）
output, dod_gate, skip_gates = model(x, t, context=context, return_gate=True)
```

---

## 兼容性说明

### 加载旧模型

由于新增了 `gate_head` 和 `skip_gates` 模块，加载旧模型时会出现 `unexpected keys` 或 `missing keys` 警告，但不影响基本功能。旧模型的权重仍然可以正常加载到对应的网络层。

### 建议的使用方式

1. **新训练**：使用新代码从头训练，门控模块会自动学习
2. **旧模型微调**：加载旧模型权重后，仅微调新增的门控模块
3. **对比实验**：设置 `model.use_gating = False` 来对比有无门控的效果

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `ldm/modules/diffusionmodules/openaimodel.py` | 修改 | 添加 SkipGate 类、gate_head、skip_gates、修改 forward |
| `evaluation_DeCo_Diff.py` | 修改 | 添加 extract_gate_maps、修改 calculate_anomaly_maps、添加命令行参数 |

---

## 技术细节

### 门控初始化

门控模块使用标准的 Xavier 初始化。由于 sigmoid 激活函数的特性，初始输出会接近 0.5，这意味着：
- 训练初期，门控对输出的影响是中性的
- 随着训练进行，门控会逐渐学会区分正常和异常区域

### 计算开销

新增的模块非常轻量：
- **gate_head**: 1 个 GroupNorm + 1 个 SiLU + 1 个 3×3 Conv + Sigmoid
- **skip_gates**: 每个跳跃连接 1 个 1×1 Conv + Sigmoid

总体增加的参数量和计算量不超过 5%。

---

## 未来改进方向

1. **联动控制**：让 Skip-Gating 的 α 受 DoD-Gating 的 g 影响
2. **多尺度门控**：在不同分辨率层使用不同的门控策略
3. **自监督预训练**：使用正常样本预训练门控模块，让其更好地识别正常模式
