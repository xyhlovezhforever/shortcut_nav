# PyTorch深度教程（六）：内置模块与实战应用

> **前置要求**：完成前五篇教程
> **核心目标**：掌握PyTorch内置模块的原理与实战应用

---

## 第一部分：nn.Module核心模块

### 1.1 nn.Linear - 全连接层

#### 1.1.1 数学原理与源码解析

```python
"""
线性层数学公式：
y = xW^T + b

其中：
- x: 输入 (batch_size, in_features)
- W: 权重矩阵 (out_features, in_features)
- b: 偏置向量 (out_features,)
- y: 输出 (batch_size, out_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class LinearLayerDeepDive:
    """全连接层深度解析"""

    def understand_linear_internals(self):
        """理解Linear层的内部实现"""

        # 创建Linear层
        linear = nn.Linear(in_features=10, out_features=5)

        # 查看参数
        print(f"权重形状: {linear.weight.shape}")  # (5, 10)
        print(f"偏置形状: {linear.bias.shape}")     # (5,)

        # 手动实现Linear层
        class MyLinear(nn.Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features

                # 初始化权重（Kaiming初始化）
                self.weight = nn.Parameter(
                    torch.randn(out_features, in_features) *
                    np.sqrt(2.0 / in_features)
                )

                if bias:
                    self.bias = nn.Parameter(torch.zeros(out_features))
                else:
                    self.register_parameter('bias', None)

            def forward(self, x):
                # y = xW^T + b
                output = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    output += self.bias
                return output

        # 测试
        my_linear = MyLinear(10, 5)
        x = torch.randn(32, 10)
        y = my_linear(x)
        print(f"输出形状: {y.shape}")  # (32, 5)

    def linear_layer_applications(self):
        """Linear层的实战应用"""

        # 1. 多层感知机（MLP）
        class MLP(nn.Module):
            """标准MLP网络"""
            def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
                super().__init__()

                layers = []
                prev_dim = input_dim

                # 隐藏层
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim

                # 输出层
                layers.append(nn.Linear(prev_dim, output_dim))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        # 2. 残差MLP
        class ResidualMLP(nn.Module):
            """带残差连接的MLP"""
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, dim)
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                residual = x
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = self.norm(x + residual)  # 残差连接
                return x

        # 3. 专家混合（Mixture of Experts）
        class MixtureOfExperts(nn.Module):
            """MoE架构"""
            def __init__(self, input_dim, hidden_dim, num_experts=4):
                super().__init__()
                self.num_experts = num_experts

                # 多个专家网络
                self.experts = nn.ModuleList([
                    nn.Linear(input_dim, hidden_dim)
                    for _ in range(num_experts)
                ])

                # 门控网络
                self.gate = nn.Linear(input_dim, num_experts)

            def forward(self, x):
                # 计算门控权重
                gate_weights = F.softmax(self.gate(x), dim=-1)

                # 计算每个专家的输出
                expert_outputs = torch.stack([
                    expert(x) for expert in self.experts
                ], dim=1)  # (batch, num_experts, hidden_dim)

                # 加权组合
                output = torch.einsum('be,beh->bh', gate_weights, expert_outputs)
                return output

### 1.2 nn.Conv2d - 卷积层

#### 1.2.1 卷积运算深度解析

```python
class ConvolutionalLayerMastery:
    """卷积层完全掌握"""

    def conv2d_mathematics(self):
        """
        卷积数学公式：

        out[b,c_out,h,w] = Σ_{c_in} Σ_{kh} Σ_{kw}
                           input[b,c_in,h*stride+kh,w*stride+kw] *
                           weight[c_out,c_in,kh,kw] +
                           bias[c_out]

        输出尺寸计算：
        H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
        W_out = floor((W_in + 2*padding - kernel_size) / stride) + 1
        """

        # 创建卷积层
        conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # 输入：(batch, channels, height, width)
        x = torch.randn(32, 3, 224, 224)
        y = conv(x)
        print(f"输出形状: {y.shape}")  # (32, 64, 224, 224)

        # 手动实现卷积（教学用，低效）
        def manual_conv2d(input, weight, bias, stride=1, padding=0):
            """手动实现2D卷积"""
            batch, in_ch, in_h, in_w = input.shape
            out_ch, _, kh, kw = weight.shape

            # 添加padding
            if padding > 0:
                input = F.pad(input, (padding,)*4)

            # 计算输出尺寸
            out_h = (in_h + 2*padding - kh) // stride + 1
            out_w = (in_w + 2*padding - kw) // stride + 1

            # 初始化输出
            output = torch.zeros(batch, out_ch, out_h, out_w)

            # 卷积计算
            for b in range(batch):
                for oc in range(out_ch):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * stride
                            w_start = w * stride

                            # 提取感受野
                            receptive_field = input[
                                b, :,
                                h_start:h_start+kh,
                                w_start:w_start+kw
                            ]

                            # 卷积
                            output[b, oc, h, w] = torch.sum(
                                receptive_field * weight[oc]
                            ) + (bias[oc] if bias is not None else 0)

            return output

    def advanced_convolution_patterns(self):
        """高级卷积模式"""

        # 1. 深度可分离卷积（Depthwise Separable Conv）
        class DepthwiseSeparableConv(nn.Module):
            """
            分解为：
            1. Depthwise Conv: 每个通道独立卷积
            2. Pointwise Conv: 1x1卷积混合通道

            参数量: in*k^2 + in*out (vs 标准卷积: in*out*k^2)
            """
            def __init__(self, in_channels, out_channels, kernel_size=3):
                super().__init__()

                # Depthwise: groups=in_channels
                self.depthwise = nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    padding=kernel_size//2, groups=in_channels
                )

                # Pointwise: 1x1卷积
                self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

            def forward(self, x):
                x = self.depthwise(x)
                x = self.pointwise(x)
                return x

        # 2. 可变形卷积（Deformable Convolution）
        class DeformableConv2d(nn.Module):
            """
            学习卷积核的空间偏移
            允许不规则感受野
            """
            def __init__(self, in_channels, out_channels, kernel_size=3):
                super().__init__()
                self.kernel_size = kernel_size

                # 标准卷积
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    padding=kernel_size//2
                )

                # 偏移预测网络
                self.offset_conv = nn.Conv2d(
                    in_channels,
                    2 * kernel_size * kernel_size,  # x和y偏移
                    kernel_size,
                    padding=kernel_size//2
                )

            def forward(self, x):
                # 预测偏移
                offset = self.offset_conv(x)

                # 应用可变形卷积（需要自定义CUDA实现）
                # 这里仅展示概念
                # output = deformable_conv2d(x, offset, self.conv.weight)

                # 简化版本：使用标准卷积
                output = self.conv(x)
                return output

        # 3. 八度卷积（Octave Convolution）
        class OctaveConv(nn.Module):
            """
            处理不同频率的特征
            高频：细节信息
            低频：全局信息
            """
            def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5):
                super().__init__()
                self.alpha = alpha

                # 高低频通道数
                in_high = int(in_channels * (1 - alpha))
                in_low = in_channels - in_high
                out_high = int(out_channels * (1 - alpha))
                out_low = out_channels - out_high

                # 高频到高频
                self.high_to_high = nn.Conv2d(
                    in_high, out_high, kernel_size, padding=kernel_size//2
                )

                # 高频到低频
                self.high_to_low = nn.Conv2d(
                    in_high, out_low, kernel_size,
                    stride=2, padding=kernel_size//2
                )

                # 低频到高频
                self.low_to_high = nn.Conv2d(
                    in_low, out_high, kernel_size, padding=kernel_size//2
                )

                # 低频到低频
                self.low_to_low = nn.Conv2d(
                    in_low, out_low, kernel_size, padding=kernel_size//2
                )

            def forward(self, x):
                # 分离高低频
                x_high, x_low = x

                # 计算各路径
                high_to_high = self.high_to_high(x_high)
                high_to_low = self.high_to_low(x_high)

                low_to_high = F.interpolate(
                    self.low_to_high(x_low),
                    size=x_high.shape[2:],
                    mode='nearest'
                )
                low_to_low = self.low_to_low(x_low)

                # 合并
                out_high = high_to_high + low_to_high
                out_low = high_to_low + low_to_low

                return out_high, out_low

### 1.3 nn.BatchNorm2d - 批归一化

#### 1.3.1 BatchNorm原理与实现

```python
class BatchNormalizationMastery:
    """批归一化完全掌握"""

    def batchnorm_mathematics(self):
        """
        BatchNorm数学公式：

        训练阶段：
        1. μ_B = (1/m) Σ x_i          # 批均值
        2. σ²_B = (1/m) Σ (x_i - μ_B)²  # 批方差
        3. x̂_i = (x_i - μ_B) / √(σ²_B + ε)  # 归一化
        4. y_i = γ * x̂_i + β          # 缩放和平移

        推理阶段：
        使用运行时统计量（指数移动平均）
        """

        # 创建BatchNorm层
        bn = nn.BatchNorm2d(num_features=64, momentum=0.1, eps=1e-5)

        # 输入
        x = torch.randn(32, 64, 56, 56)
        y = bn(x)

        # 手动实现BatchNorm
        class MyBatchNorm2d(nn.Module):
            def __init__(self, num_features, eps=1e-5, momentum=0.1):
                super().__init__()
                self.num_features = num_features
                self.eps = eps
                self.momentum = momentum

                # 可学习参数
                self.gamma = nn.Parameter(torch.ones(num_features))
                self.beta = nn.Parameter(torch.zeros(num_features))

                # 运行统计量（不参与梯度）
                self.register_buffer('running_mean', torch.zeros(num_features))
                self.register_buffer('running_var', torch.ones(num_features))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

            def forward(self, x):
                # x: (N, C, H, W)
                if self.training:
                    # 计算批统计量
                    # 在(N, H, W)维度上求平均
                    mean = x.mean(dim=(0, 2, 3), keepdim=False)
                    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=False)

                    # 更新运行统计量
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + \
                                           self.momentum * mean
                        self.running_var = (1 - self.momentum) * self.running_var + \
                                          self.momentum * var
                        self.num_batches_tracked += 1

                    # 归一化
                    x_norm = (x - mean[None, :, None, None]) / \
                             torch.sqrt(var[None, :, None, None] + self.eps)
                else:
                    # 使用运行统计量
                    x_norm = (x - self.running_mean[None, :, None, None]) / \
                             torch.sqrt(self.running_var[None, :, None, None] + self.eps)

                # 缩放和平移
                output = self.gamma[None, :, None, None] * x_norm + \
                         self.beta[None, :, None, None]

                return output

    def batchnorm_variants(self):
        """BatchNorm的变体"""

        # 1. LayerNorm（用于Transformer）
        class LayerNorm(nn.Module):
            """
            对每个样本的所有特征归一化
            不依赖batch，适合小batch或序列数据
            """
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps

                self.gamma = nn.Parameter(torch.ones(normalized_shape))
                self.beta = nn.Parameter(torch.zeros(normalized_shape))

            def forward(self, x):
                # x: (..., normalized_shape)
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, unbiased=False, keepdim=True)

                x_norm = (x - mean) / torch.sqrt(var + self.eps)
                output = self.gamma * x_norm + self.beta

                return output

        # 2. GroupNorm（介于BatchNorm和LayerNorm之间）
        class GroupNorm(nn.Module):
            """
            将通道分组，在组内归一化
            不依赖batch size
            """
            def __init__(self, num_groups, num_channels, eps=1e-5):
                super().__init__()
                assert num_channels % num_groups == 0

                self.num_groups = num_groups
                self.num_channels = num_channels
                self.eps = eps

                self.gamma = nn.Parameter(torch.ones(num_channels))
                self.beta = nn.Parameter(torch.zeros(num_channels))

            def forward(self, x):
                # x: (N, C, H, W)
                N, C, H, W = x.shape
                G = self.num_groups

                # 重塑为 (N, G, C//G, H, W)
                x = x.view(N, G, C // G, H, W)

                # 在每组内归一化
                mean = x.mean(dim=(2, 3, 4), keepdim=True)
                var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

                x_norm = (x - mean) / torch.sqrt(var + self.eps)

                # 恢复形状
                x_norm = x_norm.view(N, C, H, W)

                # 缩放和平移
                output = self.gamma[None, :, None, None] * x_norm + \
                         self.beta[None, :, None, None]

                return output

        # 3. InstanceNorm（用于风格迁移）
        class InstanceNorm(nn.Module):
            """
            对每个样本的每个通道独立归一化
            """
            def __init__(self, num_features, eps=1e-5):
                super().__init__()
                self.num_features = num_features
                self.eps = eps

                self.gamma = nn.Parameter(torch.ones(num_features))
                self.beta = nn.Parameter(torch.zeros(num_features))

            def forward(self, x):
                # x: (N, C, H, W)
                mean = x.mean(dim=(2, 3), keepdim=True)
                var = x.var(dim=(2, 3), unbiased=False, keepdim=True)

                x_norm = (x - mean) / torch.sqrt(var + self.eps)
                output = self.gamma[None, :, None, None] * x_norm + \
                         self.beta[None, :, None, None]

                return output

### 1.4 nn.Dropout - 正则化

```python
class DropoutMastery:
    """Dropout完全掌握"""

    def dropout_mathematics(self):
        """
        Dropout数学原理：

        训练阶段：
        mask ~ Bernoulli(1-p)
        output = input * mask / (1-p)  # 缩放保持期望

        推理阶段：
        output = input  # 不dropout
        """

        # 标准Dropout
        dropout = nn.Dropout(p=0.5)

        x = torch.randn(32, 128)
        y_train = dropout(x)  # 训练模式

        dropout.eval()
        y_test = dropout(x)   # 推理模式

        # 手动实现Dropout
        class MyDropout(nn.Module):
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p

            def forward(self, x):
                if not self.training:
                    return x

                # 生成mask
                mask = (torch.rand_like(x) > self.p).float()

                # 缩放
                return x * mask / (1 - self.p)

        # 测试
        my_dropout = MyDropout(0.5)
        y = my_dropout(x)

    def dropout_variants(self):
        """Dropout变体"""

        # 1. DropConnect
        class DropConnect(nn.Module):
            """
            丢弃权重而非激活
            更强的正则化
            """
            def __init__(self, linear, p=0.5):
                super().__init__()
                self.linear = linear
                self.p = p

            def forward(self, x):
                if not self.training:
                    return self.linear(x)

                # 对权重应用dropout
                weight = self.linear.weight
                mask = (torch.rand_like(weight) > self.p).float()
                dropped_weight = weight * mask / (1 - self.p)

                return F.linear(x, dropped_weight, self.linear.bias)

        # 2. Spatial Dropout（用于CNN）
        class SpatialDropout2d(nn.Module):
            """
            丢弃整个特征图
            保持空间相关性
            """
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p

            def forward(self, x):
                # x: (N, C, H, W)
                if not self.training:
                    return x

                # 生成通道级mask
                N, C, H, W = x.shape
                mask = (torch.rand(N, C, 1, 1, device=x.device) > self.p).float()

                return x * mask / (1 - self.p)

        # 3. DropBlock（更激进的Spatial Dropout）
        class DropBlock2d(nn.Module):
            """
            丢弃连续的区域块
            更有效去除语义信息
            """
            def __init__(self, p=0.1, block_size=7):
                super().__init__()
                self.p = p
                self.block_size = block_size

            def forward(self, x):
                if not self.training:
                    return x

                N, C, H, W = x.shape

                # 计算gamma（使期望丢弃率为p）
                gamma = self.p * (H * W) / (self.block_size ** 2) / \
                        ((H - self.block_size + 1) * (W - self.block_size + 1))

                # 生成mask中心点
                mask = torch.rand(N, C, H, W, device=x.device) < gamma

                # 扩展为block
                mask = F.max_pool2d(
                    mask.float(),
                    kernel_size=self.block_size,
                    stride=1,
                    padding=self.block_size // 2
                )

                mask = 1 - mask

                # 归一化
                mask = mask / mask.mean()

                return x * mask

---

## 第二部分：激活函数模块

### 2.1 经典激活函数

```python
class ActivationFunctions:
    """激活函数完全掌握"""

    def classic_activations(self):
        """经典激活函数"""

        x = torch.linspace(-5, 5, 100)

        # 1. ReLU: max(0, x)
        relu = nn.ReLU()
        y_relu = relu(x)

        # 2. Sigmoid: 1 / (1 + e^(-x))
        sigmoid = nn.Sigmoid()
        y_sigmoid = sigmoid(x)

        # 3. Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
        tanh = nn.Tanh()
        y_tanh = tanh(x)

        # 4. Leaky ReLU: max(αx, x)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        y_leaky = leaky_relu(x)

        # 手动实现
        class ManualActivations:
            @staticmethod
            def relu(x):
                return torch.maximum(x, torch.zeros_like(x))

            @staticmethod
            def sigmoid(x):
                return 1 / (1 + torch.exp(-x))

            @staticmethod
            def tanh(x):
                exp_x = torch.exp(x)
                exp_neg_x = torch.exp(-x)
                return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

            @staticmethod
            def leaky_relu(x, alpha=0.01):
                return torch.where(x > 0, x, alpha * x)

    def modern_activations(self):
        """现代激活函数"""

        # 1. GELU (Gaussian Error Linear Unit)
        class GELU(nn.Module):
            """
            GELU(x) = x * Φ(x)
            其中Φ是标准正态分布的CDF

            近似: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            """
            def forward(self, x):
                return 0.5 * x * (1 + torch.tanh(
                    np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))
                ))

        # 2. Swish / SiLU
        class Swish(nn.Module):
            """
            Swish(x) = x * σ(x)
            自门控激活函数
            """
            def forward(self, x):
                return x * torch.sigmoid(x)

        # 3. Mish
        class Mish(nn.Module):
            """
            Mish(x) = x * tanh(softplus(x))
            平滑的非单调激活
            """
            def forward(self, x):
                return x * torch.tanh(F.softplus(x))

        # 4. Hardswish（移动端优化）
        class Hardswish(nn.Module):
            """
            Hardswish(x) = x * ReLU6(x+3) / 6
            Swish的分段线性近似
            """
            def forward(self, x):
                return x * F.relu6(x + 3) / 6

        # 5. PReLU（参数化ReLU）
        prelu = nn.PReLU(num_parameters=64)  # 每个通道一个参数

        # 6. ELU (Exponential Linear Unit)
        class ELU(nn.Module):
            """
            ELU(x) = x if x > 0
                     α(e^x - 1) if x ≤ 0
            """
            def __init__(self, alpha=1.0):
                super().__init__()
                self.alpha = alpha

            def forward(self, x):
                return torch.where(
                    x > 0,
                    x,
                    self.alpha * (torch.exp(x) - 1)
                )

### 2.2 注意力机制

```python
class AttentionMechanisms:
    """注意力机制完全掌握"""

    def scaled_dot_product_attention(self):
        """缩放点积注意力"""

        class ScaledDotProductAttention(nn.Module):
            """
            Attention(Q,K,V) = softmax(QK^T / √d_k)V
            """
            def __init__(self, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(dropout)

            def forward(self, q, k, v, mask=None):
                # q,k,v: (batch, heads, seq_len, d_k)
                d_k = q.size(-1)

                # 计算注意力分数
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)

                # 应用mask
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))

                # Softmax
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # 加权求和
                output = torch.matmul(attn_weights, v)

                return output, attn_weights

    def multi_head_attention(self):
        """多头注意力"""

        class MultiHeadAttention(nn.Module):
            """
            MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
            where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
            """
            def __init__(self, d_model, num_heads, dropout=0.1):
                super().__init__()
                assert d_model % num_heads == 0

                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads

                # 线性投影
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
                self.W_o = nn.Linear(d_model, d_model)

                self.attention = ScaledDotProductAttention(dropout)
                self.dropout = nn.Dropout(dropout)

            def forward(self, q, k, v, mask=None):
                batch_size = q.size(0)

                # 线性投影并分割成多头
                q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

                # 注意力
                output, attn_weights = self.attention(q, k, v, mask)

                # 合并多头
                output = output.transpose(1, 2).contiguous().view(
                    batch_size, -1, self.d_model
                )

                # 输出投影
                output = self.W_o(output)

                return output, attn_weights

    def self_attention_variants(self):
        """自注意力变体"""

        # 1. 相对位置编码注意力
        class RelativePositionAttention(nn.Module):
            """
            加入相对位置信息
            """
            def __init__(self, d_model, num_heads, max_len=512):
                super().__init__()
                self.mha = MultiHeadAttention(d_model, num_heads)

                # 相对位置嵌入
                self.relative_pos_embedding = nn.Embedding(
                    2 * max_len - 1, d_model
                )

            def forward(self, x):
                # x: (batch, seq_len, d_model)
                seq_len = x.size(1)

                # 计算相对位置
                positions = torch.arange(seq_len, device=x.device)
                relative_positions = positions[None, :] - positions[:, None]
                relative_positions = relative_positions + seq_len - 1

                # 获取位置编码
                pos_embed = self.relative_pos_embedding(relative_positions)

                # 标准自注意力
                output, _ = self.mha(x, x, x)

                # 添加位置信息（简化版）
                output = output + pos_embed.mean(0, keepdim=True)

                return output

        # 2. 局部注意力（Local Attention）
        class LocalAttention(nn.Module):
            """
            只关注局部窗口
            降低计算复杂度
            """
            def __init__(self, d_model, num_heads, window_size=256):
                super().__init__()
                self.window_size = window_size
                self.mha = MultiHeadAttention(d_model, num_heads)

            def forward(self, x):
                # x: (batch, seq_len, d_model)
                batch_size, seq_len, d_model = x.shape

                # 分割为窗口
                num_windows = seq_len // self.window_size
                x_windows = x[:, :num_windows * self.window_size].view(
                    batch_size, num_windows, self.window_size, d_model
                )

                # 对每个窗口应用注意力
                outputs = []
                for i in range(num_windows):
                    window = x_windows[:, i]
                    output, _ = self.mha(window, window, window)
                    outputs.append(output)

                output = torch.cat(outputs, dim=1)

                # 处理剩余部分
                if seq_len % self.window_size != 0:
                    remaining = x[:, num_windows * self.window_size:]
                    remaining_out, _ = self.mha(remaining, remaining, remaining)
                    output = torch.cat([output, remaining_out], dim=1)

                return output

---

## 第三部分：损失函数

### 3.1 分类损失

```python
class ClassificationLosses:
    """分类损失函数"""

    def cross_entropy_deep_dive(self):
        """交叉熵损失深度解析"""

        # 标准交叉熵
        ce_loss = nn.CrossEntropyLoss()

        # 输入：logits (未归一化)
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        loss = ce_loss(logits, targets)

        # 手动实现
        def manual_cross_entropy(logits, targets):
            """
            CrossEntropy = -log(softmax(logits)[targets])
            """
            # Softmax
            log_probs = F.log_softmax(logits, dim=-1)

            # 负对数似然
            nll = -log_probs[range(len(targets)), targets]

            return nll.mean()

        manual_loss = manual_cross_entropy(logits, targets)
        print(f"PyTorch: {loss.item():.4f}, Manual: {manual_loss.item():.4f}")

        # 带权重的交叉熵（处理类别不平衡）
        class_weights = torch.tensor([1.0, 2.0, 3.0] + [1.0]*7)
        weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

        # Label Smoothing
        class LabelSmoothingCrossEntropy(nn.Module):
            """
            标签平滑：防止过拟合
            y_smooth = (1-ε)y_true + ε/K
            """
            def __init__(self, epsilon=0.1):
                super().__init__()
                self.epsilon = epsilon

            def forward(self, logits, targets):
                num_classes = logits.size(-1)
                log_probs = F.log_softmax(logits, dim=-1)

                # 平滑标签
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_probs)
                    true_dist.fill_(self.epsilon / (num_classes - 1))
                    true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)

                return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

    def focal_loss(self):
        """Focal Loss - 处理类别不平衡"""

        class FocalLoss(nn.Module):
            """
            FL(p_t) = -α_t(1-p_t)^γ log(p_t)

            降低易分类样本的权重
            聚焦于难分类样本
            """
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits, targets):
                # 计算交叉熵
                ce_loss = F.cross_entropy(logits, targets, reduction='none')

                # 计算p_t
                p_t = torch.exp(-ce_loss)

                # Focal权重
                focal_weight = (1 - p_t) ** self.gamma

                # 最终损失
                loss = self.alpha * focal_weight * ce_loss

                return loss.mean()

### 3.2 回归损失

```python
class RegressionLosses:
    """回归损失函数"""

    def common_regression_losses(self):
        """常见回归损失"""

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        # 1. MSE (Mean Squared Error)
        mse_loss = nn.MSELoss()
        mse = mse_loss(pred, target)

        # 2. MAE (Mean Absolute Error) / L1 Loss
        mae_loss = nn.L1Loss()
        mae = mae_loss(pred, target)

        # 3. Smooth L1 Loss (Huber Loss的变体)
        smooth_l1 = nn.SmoothL1Loss()
        loss = smooth_l1(pred, target)

        # 手动实现
        def manual_losses(pred, target):
            # MSE
            mse = torch.mean((pred - target) ** 2)

            # MAE
            mae = torch.mean(torch.abs(pred - target))

            # Smooth L1
            diff = torch.abs(pred - target)
            smooth_l1 = torch.where(
                diff < 1,
                0.5 * diff ** 2,
                diff - 0.5
            ).mean()

            return mse, mae, smooth_l1

    def huber_loss(self):
        """Huber Loss - 鲁棒回归"""

        class HuberLoss(nn.Module):
            """
            结合L1和L2的优点
            对异常值更鲁棒

            L_δ(a) = 0.5 * a^2           if |a| ≤ δ
                     δ(|a| - 0.5δ)      otherwise
            """
            def __init__(self, delta=1.0):
                super().__init__()
                self.delta = delta

            def forward(self, pred, target):
                error = torch.abs(pred - target)

                quadratic = torch.min(error, torch.tensor(self.delta))
                linear = error - quadratic

                loss = 0.5 * quadratic ** 2 + self.delta * linear

                return loss.mean()

    def quantile_loss(self):
        """分位数损失 - 预测区间"""

        class QuantileLoss(nn.Module):
            """
            用于预测分位数

            L_τ(y, ŷ) = (y - ŷ)(τ - 1_{y < ŷ})
            """
            def __init__(self, quantile=0.5):
                super().__init__()
                self.quantile = quantile

            def forward(self, pred, target):
                error = target - pred
                loss = torch.max(
                    self.quantile * error,
                    (self.quantile - 1) * error
                )
                return loss.mean()

### 3.3 对比学习损失

```python
class ContrastiveLearningLosses:
    """对比学习损失函数"""

    def contrastive_loss(self):
        """对比损失（Contrastive Loss）"""

        class ContrastiveLoss(nn.Module):
            """
            用于学习相似性度量

            L = (1-Y) * 0.5 * D^2 +
                Y * 0.5 * max(0, margin - D)^2

            其中D是欧氏距离
            """
            def __init__(self, margin=1.0):
                super().__init__()
                self.margin = margin

            def forward(self, output1, output2, label):
                # label: 1表示相似，0表示不相似
                euclidean_distance = F.pairwise_distance(output1, output2)

                loss_contrastive = torch.mean(
                    (1 - label) * torch.pow(euclidean_distance, 2) +
                    label * torch.pow(torch.clamp(
                        self.margin - euclidean_distance, min=0.0
                    ), 2)
                )

                return loss_contrastive

    def triplet_loss(self):
        """三元组损失（Triplet Loss）"""

        class TripletLoss(nn.Module):
            """
            L = max(||a-p||^2 - ||a-n||^2 + margin, 0)

            拉近anchor和positive
            推远anchor和negative
            """
            def __init__(self, margin=1.0):
                super().__init__()
                self.margin = margin

            def forward(self, anchor, positive, negative):
                pos_dist = F.pairwise_distance(anchor, positive)
                neg_dist = F.pairwise_distance(anchor, negative)

                loss = F.relu(pos_dist - neg_dist + self.margin)

                return loss.mean()

        # 三元组挖掘策略
        class TripletMiningLoss(nn.Module):
            """
            在线三元组挖掘
            选择困难样本
            """
            def __init__(self, margin=1.0, mining='hard'):
                super().__init__()
                self.margin = margin
                self.mining = mining

            def forward(self, embeddings, labels):
                # 计算所有距离
                dist_matrix = torch.cdist(embeddings, embeddings, p=2)

                # 找到positive和negative
                mask_anchor_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
                mask_anchor_negative = ~mask_anchor_positive

                if self.mining == 'hard':
                    # Hard positive: 同类中最远的
                    anchor_positive_dist = torch.where(
                        mask_anchor_positive,
                        dist_matrix,
                        torch.tensor(0.0, device=dist_matrix.device)
                    )
                    hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)

                    # Hard negative: 不同类中最近的
                    anchor_negative_dist = torch.where(
                        mask_anchor_negative,
                        dist_matrix,
                        torch.tensor(float('inf'), device=dist_matrix.device)
                    )
                    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)

                    # 三元组损失
                    loss = F.relu(
                        hardest_positive_dist - hardest_negative_dist + self.margin
                    )

                return loss.mean()

    def ntxent_loss(self):
        """NT-Xent Loss (SimCLR)"""

        class NTXentLoss(nn.Module):
            """
            归一化温度缩放交叉熵损失
            用于自监督学习
            """
            def __init__(self, temperature=0.5):
                super().__init__()
                self.temperature = temperature

            def forward(self, z_i, z_j):
                """
                z_i, z_j: (batch_size, dim) - 同一样本的两个增强视图
                """
                batch_size = z_i.size(0)

                # 归一化
                z_i = F.normalize(z_i, dim=1)
                z_j = F.normalize(z_j, dim=1)

                # 拼接
                representations = torch.cat([z_i, z_j], dim=0)

                # 计算相似度矩阵
                similarity_matrix = F.cosine_similarity(
                    representations.unsqueeze(1),
                    representations.unsqueeze(0),
                    dim=2
                )

                # 温度缩放
                similarity_matrix = similarity_matrix / self.temperature

                # 构建标签
                labels = torch.cat([
                    torch.arange(batch_size) + batch_size,
                    torch.arange(batch_size)
                ]).to(z_i.device)

                # 移除对角线
                mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
                similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

                # 交叉熵
                loss = F.cross_entropy(similarity_matrix, labels)

                return loss

---

## 第四部分：优化器深度解析

### 4.1 基础优化器

```python
class OptimizersDeepDive:
    """优化器深度解析"""

    def sgd_variants(self):
        """SGD及其变体"""

        model = nn.Linear(10, 1)

        # 1. 标准SGD
        sgd = torch.optim.SGD(model.parameters(), lr=0.01)

        # 2. SGD with Momentum
        sgd_momentum = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

        # 3. SGD with Nesterov Momentum
        sgd_nesterov = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            nesterov=True
        )

        # 手动实现SGD with Momentum
        class SGDMomentum:
            """
            v_t = β*v_{t-1} + g_t
            θ_t = θ_{t-1} - α*v_t
            """
            def __init__(self, parameters, lr=0.01, momentum=0.9):
                self.parameters = list(parameters)
                self.lr = lr
                self.momentum = momentum
                self.velocities = [
                    torch.zeros_like(p.data) for p in self.parameters
                ]

            def step(self):
                with torch.no_grad():
                    for p, v in zip(self.parameters, self.velocities):
                        if p.grad is None:
                            continue

                        # 更新速度
                        v.mul_(self.momentum).add_(p.grad)

                        # 更新参数
                        p.add_(v, alpha=-self.lr)

            def zero_grad(self):
                for p in self.parameters:
                    if p.grad is not None:
                        p.grad.zero_()

    def adam_family(self):
        """Adam系列优化器"""

        model = nn.Linear(10, 1)

        # 1. Adam
        adam = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 2. AdamW (权重衰减修正)
        adamw = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        # 3. AdamW with Lookahead
        # 需要第三方库

        # 手动实现Adam
        class AdamOptimizer:
            """
            m_t = β₁*m_{t-1} + (1-β₁)*g_t
            v_t = β₂*v_{t-1} + (1-β₂)*g_t²
            m̂_t = m_t / (1 - β₁^t)
            v̂_t = v_t / (1 - β₂^t)
            θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            """
            def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
                self.parameters = list(parameters)
                self.lr = lr
                self.beta1, self.beta2 = betas
                self.eps = eps
                self.t = 0

                self.m = [torch.zeros_like(p.data) for p in self.parameters]
                self.v = [torch.zeros_like(p.data) for p in self.parameters]

            def step(self):
                self.t += 1

                with torch.no_grad():
                    for p, m, v in zip(self.parameters, self.m, self.v):
                        if p.grad is None:
                            continue

                        # 更新矩估计
                        m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
                        v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)

                        # 偏差修正
                        bias_correction1 = 1 - self.beta1 ** self.t
                        bias_correction2 = 1 - self.beta2 ** self.t

                        m_hat = m / bias_correction1
                        v_hat = v / bias_correction2

                        # 更新参数
                        p.add_(
                            m_hat / (torch.sqrt(v_hat) + self.eps),
                            alpha=-self.lr
                        )

            def zero_grad(self):
                for p in self.parameters:
                    if p.grad is not None:
                        p.grad.zero_()

### 4.2 高级优化器

```python
class AdvancedOptimizers:
    """高级优化器"""

    def lamb_optimizer(self):
        """LAMB - Layer-wise Adaptive Moments optimizer for Batch training"""

        # 使用第三方实现或手动实现
        class LAMB:
            """
            LAMB = Adam + Layer-wise适应
            特别适合大批量训练
            """
            def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999),
                         eps=1e-6, weight_decay=0.01):
                self.parameters = list(parameters)
                self.lr = lr
                self.beta1, self.beta2 = betas
                self.eps = eps
                self.weight_decay = weight_decay
                self.t = 0

                self.m = [torch.zeros_like(p.data) for p in self.parameters]
                self.v = [torch.zeros_like(p.data) for p in self.parameters]

            def step(self):
                self.t += 1

                with torch.no_grad():
                    for p, m, v in zip(self.parameters, self.m, self.v):
                        if p.grad is None:
                            continue

                        grad = p.grad.data

                        # L2正则化
                        if self.weight_decay != 0:
                            grad = grad.add(p.data, alpha=self.weight_decay)

                        # Adam步骤
                        m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                        v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

                        m_hat = m / (1 - self.beta1 ** self.t)
                        v_hat = v / (1 - self.beta2 ** self.t)

                        adam_step = m_hat / (torch.sqrt(v_hat) + self.eps)

                        # Layer-wise适应
                        weight_norm = torch.norm(p.data)
                        adam_norm = torch.norm(adam_step)

                        if weight_norm > 0 and adam_norm > 0:
                            trust_ratio = weight_norm / adam_norm
                        else:
                            trust_ratio = 1.0

                        # 更新
                        p.add_(adam_step, alpha=-self.lr * trust_ratio)

    def lookahead_wrapper(self):
        """Lookahead优化器包装器"""

        class Lookahead:
            """
            慢权重和快权重
            周期性同步
            """
            def __init__(self, optimizer, k=5, alpha=0.5):
                self.optimizer = optimizer
                self.k = k
                self.alpha = alpha
                self.step_counter = 0

                # 保存慢权重
                self.slow_weights = [
                    p.clone().detach()
                    for group in optimizer.param_groups
                    for p in group['params']
                ]

            def step(self):
                self.optimizer.step()
                self.step_counter += 1

                if self.step_counter % self.k == 0:
                    # 更新慢权重
                    for slow_param, group in zip(
                        self.slow_weights,
                        self.optimizer.param_groups
                    ):
                        for fast_param in group['params']:
                            slow_param.data.add_(
                                fast_param.data - slow_param.data,
                                alpha=self.alpha
                            )
                            fast_param.data.copy_(slow_param.data)

        # 使用示例
        model = nn.Linear(10, 1)
        base_optimizer = torch.optim.Adam(model.parameters())
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

---

## 第五部分：实战案例

### 5.1 图像分类完整流程

```python
class ImageClassificationPipeline:
    """图像分类完整流程"""

    def build_resnet_from_scratch(self):
        """从零构建ResNet"""

        class BasicBlock(nn.Module):
            """ResNet基础块"""
            expansion = 1

            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()

                self.conv1 = nn.Conv2d(
                    in_channels, out_channels, 3,
                    stride=stride, padding=1, bias=False
                )
                self.bn1 = nn.BatchNorm2d(out_channels)

                self.conv2 = nn.Conv2d(
                    out_channels, out_channels, 3,
                    padding=1, bias=False
                )
                self.bn2 = nn.BatchNorm2d(out_channels)

                # 捷径连接
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, 1,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(out_channels)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out

        class ResNet(nn.Module):
            """ResNet完整网络"""
            def __init__(self, block, num_blocks, num_classes=10):
                super().__init__()
                self.in_channels = 64

                # 初始层
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)

                # 残差层
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

                # 分类头
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            def _make_layer(self, block, out_channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []

                for stride in strides:
                    layers.append(block(self.in_channels, out_channels, stride))
                    self.in_channels = out_channels * block.expansion

                return nn.Sequential(*layers)

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out

        # 创建ResNet-18
        def ResNet18(num_classes=10):
            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

        model = ResNet18(num_classes=10)
        return model

    def complete_training_loop(self):
        """完整训练循环"""

        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        # 1. 数据准备
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

        # 2. 加载数据集
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=128,
            shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=100,
            shuffle=False, num_workers=2
        )

        # 3. 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.build_resnet_from_scratch().to(device)

        # 4. 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200
        )

        # 5. 训练循环
        def train_epoch(epoch):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                          f'Loss: {train_loss/(batch_idx+1):.3f} '
                          f'Acc: {100.*correct/total:.3f}%')

            return train_loss / len(trainloader), 100. * correct / total

        # 6. 验证循环
        def validate():
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            print(f'Test Loss: {test_loss/len(testloader):.3f} '
                  f'Test Acc: {acc:.3f}%')

            return test_loss / len(testloader), acc

        # 7. 主训练循环
        num_epochs = 200
        best_acc = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch}')

            train_loss, train_acc = train_epoch(epoch)
            test_loss, test_acc = validate()
            scheduler.step()

            # 保存最佳模型
            if test_acc > best_acc:
                print('Saving...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': test_acc,
                }, 'best_model.pth')
                best_acc = test_acc

### 5.2 文本分类（Transformer）

```python
class TextClassificationTransformer:
    """基于Transformer的文本分类"""

    def build_transformer_classifier(self):
        """构建Transformer分类器"""

        class TransformerEncoder(nn.Module):
            """Transformer编码器"""
            def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
                super().__init__()

                # 多头注意力
                self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
                self.norm1 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)

                # 前馈网络
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                )
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout2 = nn.Dropout(dropout)

            def forward(self, x, mask=None):
                # 自注意力
                attn_output, _ = self.self_attn(x, x, x, mask)
                x = self.norm1(x + self.dropout1(attn_output))

                # 前馈网络
                ffn_output = self.ffn(x)
                x = self.norm2(x + self.dropout2(ffn_output))

                return x

        class TextClassifier(nn.Module):
            """完整的文本分类器"""
            def __init__(self, vocab_size, d_model=512, num_heads=8,
                         num_layers=6, d_ff=2048, max_len=512,
                         num_classes=2, dropout=0.1):
                super().__init__()

                # 词嵌入
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(
                    torch.randn(1, max_len, d_model)
                )
                self.dropout = nn.Dropout(dropout)

                # Transformer编码器层
                self.encoder_layers = nn.ModuleList([
                    TransformerEncoder(d_model, num_heads, d_ff, dropout)
                    for _ in range(num_layers)
                ])

                # 分类头
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, num_classes)
                )

            def forward(self, x, mask=None):
                # x: (batch, seq_len)
                seq_len = x.size(1)

                # 嵌入 + 位置编码
                x = self.embedding(x) * np.sqrt(self.d_model)
                x = x + self.pos_encoding[:, :seq_len]
                x = self.dropout(x)

                # Transformer编码
                for layer in self.encoder_layers:
                    x = layer(x, mask)

                # 池化 (取[CLS]或平均)
                x = x.mean(dim=1)

                # 分类
                logits = self.classifier(x)

                return logits

---

## 总结

本教程详细讲解了PyTorch内置模块的原理与实战应用：

### 核心模块
- ✅ nn.Linear - 全连接层
- ✅ nn.Conv2d - 卷积层
- ✅ nn.BatchNorm2d - 批归一化
- ✅ nn.Dropout - 正则化

### 激活函数
- ✅ 经典激活（ReLU, Sigmoid, Tanh）
- ✅ 现代激活（GELU, Swish, Mish）
- ✅ 注意力机制

### 损失函数
- ✅ 分类损失（CrossEntropy, Focal）
- ✅ 回归损失（MSE, Huber, Quantile）
- ✅ 对比学习损失（Contrastive, Triplet, NT-Xent)

### 优化器
- ✅ 基础优化器（SGD, Adam）
- ✅ 高级优化器（LAMB, Lookahead）

### 实战案例
- ✅ 图像分类（ResNet）
- ✅ 文本分类（Transformer）

掌握这些内置模块，你就能构建任何深度学习模型！
