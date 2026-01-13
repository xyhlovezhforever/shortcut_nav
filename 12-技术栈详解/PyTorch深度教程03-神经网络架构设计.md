# PyTorch深度教程（三）：神经网络架构设计

> **前置要求**：完成教程一和教程二
> **核心目标**：深度理解神经网络的数学原理与工程设计

---

## 第一部分：神经网络的数学基础

### 1.1 通用逼近定理

#### 1.1.1 数学定理与证明思路

```python
"""
通用逼近定理（Universal Approximation Theorem）

定理：
对于任意连续函数 f: [0,1]ⁿ → ℝ 和任意 ε > 0，
存在单隐层神经网络 g(x) = Σᵢ αᵢ σ(wᵢᵀx + bᵢ)，
使得 sup_{x∈[0,1]ⁿ} |f(x) - g(x)| < ε

其中 σ 是非多项式的有界连续激活函数
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class UniversalApproximation:
    """通用逼近定理的实验验证"""

    def approximate_sine_wave(self):
        """
        用单隐层网络逼近sin函数
        """
        class SingleHiddenLayerNet(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(1, hidden_size)
                self.fc2 = nn.Linear(hidden_size, 1)

            def forward(self, x):
                h = torch.tanh(self.fc1(x))
                return self.fc2(h)

        # 训练数据
        x_train = torch.linspace(0, 2*np.pi, 1000).unsqueeze(1)
        y_train = torch.sin(x_train)

        # 不同隐层大小的网络
        hidden_sizes = [10, 50, 200]

        for hidden_size in hidden_sizes:
            model = SingleHiddenLayerNet(hidden_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            # 训练
            for epoch in range(1000):
                optimizer.zero_grad()
                output = model(x_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()

            # 评估
            with torch.no_grad():
                y_pred = model(x_train)
                error = torch.abs(y_pred - y_train).max()
                print(f"隐层大小={hidden_size}, 最大误差={error:.6f}")

    def visualize_feature_space(self):
        """
        可视化隐层特征空间
        理解网络如何分解复杂函数
        """
        model = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # 创建网格
        x = torch.linspace(-5, 5, 100)
        y = torch.linspace(-5, 5, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

        # 前向传播到隐层
        with torch.no_grad():
            hidden = torch.relu(model[0](grid))

            # 分析隐层激活模式
            print(f"激活的神经元比例: {(hidden > 0).float().mean():.2%}")
```

### 1.2 激活函数的数学性质

#### 1.2.1 常见激活函数分析

```python
class ActivationFunctionAnalysis:
    """激活函数的数学分析"""

    def activation_properties(self):
        """
        关键性质：
        1. 非线性性
        2. 可微性
        3. 单调性
        4. 值域
        5. 梯度饱和
        """

        x = torch.linspace(-5, 5, 1000)

        # 1. Sigmoid: σ(x) = 1/(1+e^(-x))
        sigmoid = torch.sigmoid(x)
        sigmoid_grad = sigmoid * (1 - sigmoid)  # σ'(x) = σ(x)(1-σ(x))

        # 2. Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
        tanh = torch.tanh(x)
        tanh_grad = 1 - tanh**2  # tanh'(x) = 1 - tanh²(x)

        # 3. ReLU: max(0, x)
        relu = torch.relu(x)
        relu_grad = (x > 0).float()  # 分段导数

        # 4. Leaky ReLU: max(αx, x)
        leaky_relu = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

        # 5. GELU (Gaussian Error Linear Unit)
        # GELU(x) = x·Φ(x), Φ是标准正态分布的CDF
        gelu = torch.nn.functional.gelu(x)

        # 6. Swish/SiLU: x·σ(x)
        swish = x * torch.sigmoid(x)
        swish_grad = torch.sigmoid(x) + x * sigmoid * (1 - sigmoid)

        # 7. Mish: x·tanh(softplus(x))
        mish = x * torch.tanh(torch.nn.functional.softplus(x))

    def analyze_gradient_flow(self):
        """
        分析梯度流动特性
        """
        # 梯度饱和问题
        def gradient_saturation_test(activation_fn, name):
            x = torch.linspace(-10, 10, 1000, requires_grad=True)
            y = activation_fn(x)
            y.sum().backward()

            # 分析梯度大小
            grad_norm = x.grad.abs()
            saturated_ratio = (grad_norm < 0.01).float().mean()

            print(f"{name}:")
            print(f"  饱和区域比例: {saturated_ratio:.2%}")
            print(f"  平均梯度: {grad_norm.mean():.6f}")

        # 测试各种激活函数
        gradient_saturation_test(torch.sigmoid, "Sigmoid")
        gradient_saturation_test(torch.tanh, "Tanh")
        gradient_saturation_test(torch.relu, "ReLU")
        gradient_saturation_test(torch.nn.functional.gelu, "GELU")

    def custom_activation_functions(self):
        """
        自定义激活函数
        """
        # 1. Parametric ReLU
        class PReLU(nn.Module):
            def __init__(self, num_parameters=1, init=0.25):
                super().__init__()
                self.alpha = nn.Parameter(torch.ones(num_parameters) * init)

            def forward(self, x):
                return torch.where(x > 0, x, self.alpha * x)

        # 2. Adaptive Piecewise Linear (APL)
        class APL(nn.Module):
            def __init__(self, num_segments=3):
                super().__init__()
                self.num_segments = num_segments
                self.slopes = nn.Parameter(torch.ones(num_segments))
                self.biases = nn.Parameter(torch.zeros(num_segments))

            def forward(self, x):
                # 分段线性激活
                output = torch.zeros_like(x)
                for i in range(self.num_segments):
                    mask = (x >= i) & (x < i+1)
                    output += mask.float() * (self.slopes[i] * x + self.biases[i])
                return output

        # 3. Smooth ReLU (Softplus)
        def smooth_relu(x, beta=1):
            """
            softplus(x) = (1/β)·log(1 + e^(βx))
            当β→∞时，收敛到ReLU
            """
            return torch.nn.functional.softplus(x, beta=beta)
```

### 1.3 损失函数的数学理论

#### 1.3.1 损失函数与概率分布

```python
class LossFunctionTheory:
    """损失函数的数学基础"""

    def maximum_likelihood_perspective(self):
        """
        从最大似然估计看损失函数

        给定模型 p(y|x;θ)，最大似然估计：
        θ* = argmax Π p(yᵢ|xᵢ;θ)
           = argmax Σ log p(yᵢ|xᵢ;θ)
           = argmin Σ -log p(yᵢ|xᵢ;θ)
        """

        # 1. 均方误差 ↔ 高斯分布
        # 假设 y ~ N(f(x), σ²)
        # -log p(y|x) ∝ (y - f(x))²
        def mse_as_gaussian_nll(predictions, targets, sigma=1.0):
            """MSE作为高斯负对数似然"""
            nll = 0.5 * torch.log(2 * np.pi * sigma**2) + \
                  (targets - predictions)**2 / (2 * sigma**2)
            return nll.mean()

        # 2. 交叉熵 ↔ 伯努利分布
        # 假设 y ~ Bernoulli(p)
        # -log p(y|x) = -y·log(p) - (1-y)·log(1-p)
        def binary_cross_entropy_derivation(predictions, targets):
            """二元交叉熵推导"""
            # predictions是概率 p
            bce = -(targets * torch.log(predictions) +
                    (1 - targets) * torch.log(1 - predictions))
            return bce.mean()

        # 3. 交叉熵 ↔ 类别分布
        # 假设 y ~ Categorical(p₁, ..., pₖ)
        # -log p(y=c|x) = -log pᶜ
        def categorical_cross_entropy_derivation(logits, targets):
            """多类交叉熵推导"""
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[range(len(targets)), targets]
            return nll.mean()

    def loss_function_properties(self):
        """
        损失函数的关键性质
        """

        # 1. 凸性
        def check_convexity_mse():
            """
            MSE是凸函数
            对于线性模型 f(x) = wx + b
            L(w) = Σ(y - wx)² 是w的凸函数
            """
            w = torch.linspace(-5, 5, 100, requires_grad=True)
            x = torch.tensor(2.0)
            y = torch.tensor(3.0)

            losses = []
            for w_val in w:
                loss = (y - w_val * x) ** 2
                losses.append(loss.item())

            # Hessian为正 → 凸
            # d²L/dw² = 2x² > 0

        # 2. Lipschitz连续性
        # |L(θ₁) - L(θ₂)| ≤ K·||θ₁ - θ₂||
        # 保证梯度有界

        # 3. 对称性
        def symmetric_loss(pred, target):
            """对称损失：L(a,b) = L(b,a)"""
            return (pred - target) ** 2  # MSE是对称的

        def asymmetric_loss(pred, target, alpha=0.5):
            """非对称损失：用于不平衡数据"""
            error = target - pred
            return torch.where(
                error >= 0,
                alpha * error ** 2,
                (1 - alpha) * error ** 2
            )

    def advanced_loss_functions(self):
        """
        高级损失函数
        """

        # 1. Focal Loss（处理类别不平衡）
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                """
                FL(pₜ) = -αₜ(1-pₜ)^γ log(pₜ)
                降低易分类样本的权重
                """
                ce_loss = nn.functional.cross_entropy(
                    inputs, targets, reduction='none'
                )
                p_t = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
                return focal_loss.mean()

        # 2. Contrastive Loss（对比学习）
        class ContrastiveLoss(nn.Module):
            def __init__(self, temperature=0.5):
                super().__init__()
                self.temperature = temperature

            def forward(self, features, labels):
                """
                SimCLR风格的对比损失
                """
                # 归一化特征
                features = nn.functional.normalize(features, dim=1)

                # 计算相似度矩阵
                similarity_matrix = torch.matmul(features, features.T)
                similarity_matrix = similarity_matrix / self.temperature

                # 构建正样本掩码
                batch_size = features.shape[0]
                mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)

                # 计算对比损失
                positives = similarity_matrix[mask].view(batch_size, -1)
                negatives = similarity_matrix[~mask].view(batch_size, -1)

                logits = torch.cat([positives, negatives], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)

                loss = nn.functional.cross_entropy(logits, labels)
                return loss

        # 3. Triplet Loss（度量学习）
        class TripletLoss(nn.Module):
            def __init__(self, margin=1.0):
                super().__init__()
                self.margin = margin

            def forward(self, anchor, positive, negative):
                """
                L = max(||a-p||² - ||a-n||² + margin, 0)
                """
                pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
                neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

                loss = torch.relu(pos_dist - neg_dist + self.margin)
                return loss.mean()

        # 4. Huber Loss（鲁棒回归）
        def huber_loss(pred, target, delta=1.0):
            """
            结合L1和L2的优点
            对异常值更鲁棒
            """
            abs_error = torch.abs(pred - target)
            quadratic = torch.min(abs_error, torch.tensor(delta))
            linear = abs_error - quadratic
            return (0.5 * quadratic ** 2 + delta * linear).mean()
```

---

## 第二部分：网络架构组件

### 2.1 全连接层的深度分析

#### 2.1.1 全连接层的数学本质

```python
class FullyConnectedLayerAnalysis:
    """全连接层的深度解析"""

    def mathematical_formulation(self):
        """
        数学表达：
        y = σ(Wx + b)

        其中：
        - W ∈ ℝ^(out×in): 权重矩阵
        - b ∈ ℝ^out: 偏置向量
        - σ: 激活函数
        """

        # 从零实现全连接层
        class LinearLayer(nn.Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features

                # Xavier初始化
                self.weight = nn.Parameter(
                    torch.randn(out_features, in_features) *
                    np.sqrt(2.0 / (in_features + out_features))
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

            def extra_repr(self):
                return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def geometric_interpretation(self):
        """
        几何解释：全连接层作为线性变换
        """
        # 权重矩阵的奇异值分解
        W = torch.randn(3, 2)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # W = U @ diag(S) @ V^T
        # 三个几何变换的组合：
        # 1. V^T: 旋转/反射
        # 2. diag(S): 缩放
        # 3. U: 旋转/反射

        # 可视化：2D -> 3D变换
        x = torch.randn(100, 2)  # 100个2D点
        y = x @ W.t()            # 变换到3D

        print(f"输入空间: {x.shape}")
        print(f"输出空间: {y.shape}")
        print(f"奇异值: {S}")  # 表示每个方向的缩放因子

    def capacity_analysis(self):
        """
        表达能力分析
        """
        # 参数量
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 单层网络
        model1 = nn.Linear(100, 10)
        params1 = count_parameters(model1)
        print(f"单层参数量: {params1}")  # 100*10 + 10 = 1010

        # 多层网络
        model2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        params2 = count_parameters(model2)
        print(f"两层参数量: {params2}")  # 100*50 + 50 + 50*10 + 10 = 5560

        # VC维（Vapnik-Chervonenkis维）
        # 单层感知器：d+1（d是输入维度）
        # 深度网络：指数级增长
```

### 2.2 卷积层的数学原理

#### 2.2.1 卷积的数学定义

```python
class ConvolutionalLayerMathematics:
    """卷积层的数学分析"""

    def convolution_definition(self):
        """
        连续情况：
        (f * g)(t) = ∫ f(τ)g(t-τ) dτ

        离散情况（1D）：
        (f * g)[n] = Σₘ f[m]g[n-m]

        2D图像卷积：
        (I * K)[i,j] = ΣₘΣₙ I[i-m, j-n]K[m,n]
        """

        # 从零实现2D卷积
        def conv2d_naive(input, kernel, stride=1, padding=0):
            """
            朴素实现（教学用）
            input: (batch, in_channels, height, width)
            kernel: (out_channels, in_channels, kH, kW)
            """
            batch, in_ch, in_h, in_w = input.shape
            out_ch, _, kH, kW = kernel.shape

            # 添加padding
            if padding > 0:
                input = torch.nn.functional.pad(
                    input, (padding, padding, padding, padding)
                )

            # 计算输出尺寸
            out_h = (in_h + 2*padding - kH) // stride + 1
            out_w = (in_w + 2*padding - kW) // stride + 1

            # 输出张量
            output = torch.zeros(batch, out_ch, out_h, out_w)

            # 卷积计算
            for b in range(batch):
                for oc in range(out_ch):
                    for i in range(out_h):
                        for j in range(out_w):
                            # 提取感受野
                            h_start = i * stride
                            w_start = j * stride
                            receptive_field = input[
                                b, :,
                                h_start:h_start+kH,
                                w_start:w_start+kW
                            ]

                            # 计算卷积
                            output[b, oc, i, j] = torch.sum(
                                receptive_field * kernel[oc]
                            )

            return output

    def convolution_as_matrix_multiplication(self):
        """
        卷积作为矩阵乘法（Toeplitz矩阵）
        """
        # im2col技巧
        def im2col(input, kernel_size, stride=1, padding=0):
            """
            将图像展开为列矩阵
            每列包含一个感受野
            """
            batch, channels, height, width = input.shape
            kH, kW = kernel_size

            # 添加padding
            input_padded = torch.nn.functional.pad(
                input, (padding, padding, padding, padding)
            )

            # 输出尺寸
            out_h = (height + 2*padding - kH) // stride + 1
            out_w = (width + 2*padding - kW) // stride + 1

            # 使用unfold展开
            # (batch, C*kH*kW, out_h*out_w)
            col = torch.nn.functional.unfold(
                input_padded,
                kernel_size=(kH, kW),
                stride=stride
            )

            return col, (out_h, out_w)

        # 使用矩阵乘法进行卷积
        def conv2d_via_matmul(input, kernel, stride=1, padding=0):
            """
            通过矩阵乘法实现卷积
            """
            batch, in_ch, in_h, in_w = input.shape
            out_ch, _, kH, kW = kernel.shape

            # im2col
            col, (out_h, out_w) = im2col(
                input, (kH, kW), stride, padding
            )

            # 重塑kernel: (out_ch, in_ch*kH*kW)
            kernel_reshaped = kernel.view(out_ch, -1)

            # 矩阵乘法: (out_ch, in_ch*kH*kW) @ (in_ch*kH*kW, out_h*out_w)
            # = (out_ch, out_h*out_w)
            output = kernel_reshaped @ col

            # 重塑输出
            output = output.view(batch, out_ch, out_h, out_w)

            return output

    def advanced_convolution_types(self):
        """
        高级卷积类型
        """

        # 1. 转置卷积（反卷积）
        class TransposedConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size, stride=2, padding=1
                )

            def forward(self, x):
                """
                上采样：增加空间分辨率
                """
                return self.conv(x)

        # 2. 空洞卷积（Dilated Convolution）
        class DilatedConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    dilation=dilation, padding=dilation
                )

            def forward(self, x):
                """
                扩大感受野，不增加参数
                有效kernel大小 = (kernel_size-1) * dilation + 1
                """
                return self.conv(x)

        # 3. 深度可分离卷积（Depthwise Separable）
        class DepthwiseSeparableConv(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super().__init__()
                # Depthwise: 每个输入通道独立卷积
                self.depthwise = nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    groups=in_channels, padding=kernel_size//2
                )

                # Pointwise: 1x1卷积混合通道
                self.pointwise = nn.Conv2d(
                    in_channels, out_channels, 1
                )

            def forward(self, x):
                x = self.depthwise(x)
                x = self.pointwise(x)
                return x

            def parameter_reduction(self):
                """
                参数量对比：
                标准卷积: in_ch × out_ch × k × k
                深度可分离: in_ch × k × k + in_ch × out_ch
                减少比例 ≈ 1/out_ch + 1/k²
                """
                pass

        # 4. 分组卷积（Group Convolution）
        class GroupConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, groups):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    groups=groups, padding=kernel_size//2
                )

            def forward(self, x):
                """
                将输入输出通道分组
                减少参数: 原来的1/groups
                """
                return self.conv(x)
```

### 2.3 归一化层的理论

#### 2.3.1 批归一化（Batch Normalization）

```python
class NormalizationLayerTheory:
    """归一化层的数学原理"""

    def batch_normalization_math(self):
        """
        Batch Normalization数学公式：

        1. 计算批统计量：
           μ_B = (1/m) Σᵢ xᵢ
           σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²

        2. 归一化：
           x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

        3. 缩放和平移：
           yᵢ = γx̂ᵢ + β

        其中γ和β是可学习参数
        """

        class BatchNorm1dFromScratch(nn.Module):
            def __init__(self, num_features, eps=1e-5, momentum=0.1):
                super().__init__()
                self.num_features = num_features
                self.eps = eps
                self.momentum = momentum

                # 可学习参数
                self.gamma = nn.Parameter(torch.ones(num_features))
                self.beta = nn.Parameter(torch.zeros(num_features))

                # 运行时统计量（不参与反向传播）
                self.register_buffer('running_mean', torch.zeros(num_features))
                self.register_buffer('running_var', torch.ones(num_features))

            def forward(self, x):
                """
                x: (batch_size, num_features)
                """
                if self.training:
                    # 训练模式：使用批统计量
                    batch_mean = x.mean(dim=0)
                    batch_var = x.var(dim=0, unbiased=False)

                    # 归一化
                    x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

                    # 更新运行统计量
                    self.running_mean = (1 - self.momentum) * self.running_mean + \
                                        self.momentum * batch_mean
                    self.running_var = (1 - self.momentum) * self.running_var + \
                                       self.momentum * batch_var
                else:
                    # 推理模式：使用运行统计量
                    x_norm = (x - self.running_mean) / \
                             torch.sqrt(self.running_var + self.eps)

                # 缩放和平移
                output = self.gamma * x_norm + self.beta
                return output

    def layer_normalization(self):
        """
        Layer Normalization
        对每个样本的所有特征归一化
        """
        class LayerNormFromScratch(nn.Module):
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps

                self.gamma = nn.Parameter(torch.ones(normalized_shape))
                self.beta = nn.Parameter(torch.zeros(normalized_shape))

            def forward(self, x):
                """
                x: (..., normalized_shape)
                对最后的normalized_shape维度归一化
                """
                # 计算统计量
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)

                # 归一化
                x_norm = (x - mean) / torch.sqrt(var + self.eps)

                # 缩放和平移
                output = self.gamma * x_norm + self.beta
                return output

    def comparison_of_normalizations(self):
        """
        不同归一化方法的对比
        """
        # 创建测试数据: (batch, channels, height, width)
        x = torch.randn(4, 3, 8, 8)

        # 1. Batch Norm: 对(batch, height, width)归一化
        bn = nn.BatchNorm2d(3)
        y_bn = bn(x)
        print(f"BN: 每个通道的均值接近0, std接近1")

        # 2. Layer Norm: 对(channels, height, width)归一化
        ln = nn.LayerNorm([3, 8, 8])
        y_ln = ln(x)
        print(f"LN: 每个样本的均值接近0, std接近1")

        # 3. Instance Norm: 对(height, width)归一化
        inorm = nn.InstanceNorm2d(3)
        y_in = inorm(x)
        print(f"IN: 每个样本每个通道的均值接近0, std接近1")

        # 4. Group Norm: 通道分组归一化
        gn = nn.GroupNorm(num_groups=1, num_channels=3)  # groups=1相当于LayerNorm
        y_gn = gn(x)

        """
        应用场景：
        - Batch Norm: CNN, 大批量
        - Layer Norm: Transformer, RNN, 小批量
        - Instance Norm: 风格迁移
        - Group Norm: 小批量的CNN
        """

    def why_normalization_works(self):
        """
        归一化为什么有效？
        """
        # 1. 减少内部协变量偏移（Internal Covariate Shift）
        # 层输入分布稳定，加速训练

        # 2. 平滑损失函数
        # 使损失曲面更平滑，梯度更稳定

        # 3. 允许更大的学习率

        # 4. 正则化效果
        # BN引入噪声（批统计量的随机性）

        # 验证：损失曲面的Lipschitz常数
        def compute_lipschitz_constant(model, x):
            """
            估计梯度的Lipschitz常数
            """
            x.requires_grad_(True)
            y = model(x)
            y.sum().backward()
            grad_norm = x.grad.norm()
            return grad_norm.item()

        # 对比有无BN的模型
        model_without_bn = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        model_with_bn = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        x = torch.randn(32, 100)
        lip_without = compute_lipschitz_constant(model_without_bn, x)
        lip_with = compute_lipschitz_constant(model_with_bn, x)

        print(f"Lipschitz常数（无BN）: {lip_without:.4f}")
        print(f"Lipschitz常数（有BN）: {lip_with:.4f}")
```

---

## 第三部分：现代网络架构

### 3.1 残差网络（ResNet）

```python
class ResidualNetworkArchitecture:
    """残差网络的设计与分析"""

    def residual_block_theory(self):
        """
        残差连接的数学原理

        标准映射：H(x) = F(x)
        残差映射：H(x) = F(x) + x

        优势：
        1. 梯度直接流动：∂H/∂x = ∂F/∂x + 1
        2. 恒等映射容易学习：F(x) = 0
        3. 深度网络不退化
        """

        class BasicBlock(nn.Module):
            """ResNet基础块"""
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()

                self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                                        stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)

                self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                        padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)

                # 捷径连接
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1,
                                  stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )

            def forward(self, x):
                # 主路径
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))

                # 残差连接
                out += self.shortcut(x)
                out = torch.relu(out)

                return out

        class BottleneckBlock(nn.Module):
            """ResNet瓶颈块（ResNet-50+）"""
            expansion = 4

            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()

                # 1x1降维
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)

                # 3x3卷积
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                        stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)

                # 1x1升维
                self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                                        1, bias=False)
                self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels * self.expansion:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels * self.expansion,
                                  1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels * self.expansion)
                    )

            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = torch.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                out += self.shortcut(x)
                out = torch.relu(out)
                return out

    def gradient_flow_analysis(self):
        """
        梯度流动分析
        """
        # 反向传播公式
        # ∂L/∂x_l = ∂L/∂x_L · ∂x_L/∂x_l
        #          = ∂L/∂x_L · (1 + ∂F/∂x_l)

        # 对比：普通网络的梯度
        # ∂L/∂x_l = ∂L/∂x_L · Π_{i=l}^{L-1} ∂F_i/∂x_i

        def visualize_gradient_magnitudes():
            """可视化不同深度的梯度大小"""
            # 普通网络
            plain_net = nn.Sequential(*[
                nn.Linear(100, 100) for _ in range(50)
            ])

            # 残差网络（简化）
            class SimpleResNet(nn.Module):
                def __init__(self, depth):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        nn.Linear(100, 100) for _ in range(depth)
                    ])

                def forward(self, x):
                    for layer in self.layers:
                        x = x + layer(x)  # 残差连接
                    return x

            res_net = SimpleResNet(50)

            # 测试梯度
            x = torch.randn(1, 100, requires_grad=True)

            # 普通网络
            y1 = plain_net(x)
            y1.sum().backward()
            grad_plain = x.grad.norm()

            # 残差网络
            x.grad = None
            y2 = res_net(x)
            y2.sum().backward()
            grad_res = x.grad.norm()

            print(f"普通网络梯度: {grad_plain:.6f}")
            print(f"残差网络梯度: {grad_res:.6f}")
```

### 3.2 注意力机制

```python
class AttentionMechanisms:
    """注意力机制的数学与实现"""

    def scaled_dot_product_attention(self):
        """
        缩放点积注意力

        Attention(Q, K, V) = softmax(QK^T / √d_k) V

        其中：
        - Q: 查询 (queries)
        - K: 键 (keys)
        - V: 值 (values)
        - d_k: 键的维度
        """

        def attention(Q, K, V, mask=None):
            """
            Q: (batch, num_heads, seq_len_q, d_k)
            K: (batch, num_heads, seq_len_k, d_k)
            V: (batch, num_heads, seq_len_v, d_v)
            """
            d_k = Q.shape[-1]

            # 计算注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

            # 应用掩码
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # 加权求和
            output = torch.matmul(attn_weights, V)

            return output, attn_weights

    def multi_head_attention_math(self):
        """
        多头注意力的数学原理

        MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O

        其中 head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
        """

        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, num_heads, dropout=0.1):
                super().__init__()
                assert d_model % num_heads == 0

                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads

                # 投影矩阵
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
                self.W_o = nn.Linear(d_model, d_model)

                self.dropout = nn.Dropout(dropout)

            def forward(self, query, key, value, mask=None):
                batch_size = query.shape[0]

                # 线性投影并分割成多头
                Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

                # 缩放点积注意力
                d_k = Q.shape[-1]
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))

                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # 加权求和
                context = torch.matmul(attn_weights, V)

                # 合并多头
                context = context.transpose(1, 2).contiguous().view(
                    batch_size, -1, self.d_model
                )

                # 输出投影
                output = self.W_o(context)

                return output, attn_weights

    def self_attention_variants(self):
        """
        自注意力的变体
        """

        # 1. 相对位置编码
        class RelativePositionAttention(nn.Module):
            def __init__(self, d_model, num_heads, max_len=512):
                super().__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads

                # 相对位置嵌入
                self.relative_positions = nn.Parameter(
                    torch.randn(2 * max_len - 1, self.d_k)
                )

            def forward(self, Q, K, V):
                """
                添加相对位置信息到注意力分数
                """
                # 标准注意力分数
                scores_content = torch.matmul(Q, K.transpose(-2, -1))

                # 相对位置分数
                seq_len = Q.shape[2]
                # 计算相对位置分数...

                scores = (scores_content + 0) / np.sqrt(self.d_k)
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, V)

                return output

        # 2. 线性注意力（Linear Attention）
        def linear_attention(Q, K, V):
            """
            使用核技巧避免显式计算注意力矩阵
            复杂度: O(N) 而非 O(N²)
            """
            # 应用激活函数（如elu+1）
            Q = torch.nn.functional.elu(Q) + 1
            K = torch.nn.functional.elu(K) + 1

            # K^T V: (d_k, d_v)
            KV = torch.matmul(K.transpose(-2, -1), V)

            # Q(K^T V): (seq_len_q, d_v)
            output = torch.matmul(Q, KV)

            # 归一化
            normalizer = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1))
            output = output / (normalizer + 1e-6)

            return output
```

---

## 第四部分：网络设计原则

### 4.1 深度与宽度的权衡

```python
class DepthWidthTradeoff:
    """深度与宽度的权衡"""

    def parameter_budget_analysis(self):
        """
        给定参数预算，如何分配深度和宽度？
        """

        def create_model(depth, width, input_dim=100, output_dim=10):
            """创建指定深度和宽度的模型"""
            layers = [nn.Linear(input_dim, width), nn.ReLU()]

            for _ in range(depth - 2):
                layers.extend([nn.Linear(width, width), nn.ReLU()])

            layers.append(nn.Linear(width, output_dim))

            return nn.Sequential(*layers)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        # 固定参数量，尝试不同的深度-宽度组合
        target_params = 50000

        configs = [
            (5, 100),   # 深度5, 宽度100
            (10, 70),   # 深度10, 宽度70
            (20, 50),   # 深度20, 宽度50
        ]

        for depth, width in configs:
            model = create_model(depth, width)
            params = count_parameters(model)
            print(f"深度={depth}, 宽度={width}: {params} 参数")

            # 经验规律：
            # - 更深的网络：更强的表达能力，但更难训练
            # - 更宽的网络：更容易训练，但可能过拟合

    def effective_depth_analysis(self):
        """
        有效深度分析
        """

        class DynamicDepthNet(nn.Module):
            """
            带有动态深度的网络
            每层有概率被跳过（Stochastic Depth）
            """
            def __init__(self, depth, width, survival_prob=0.8):
                super().__init__()
                self.depth = depth
                self.survival_prob = survival_prob

                self.layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(width, width),
                        nn.ReLU()
                    )
                    for _ in range(depth)
                ])

            def forward(self, x):
                for layer in self.layers:
                    if self.training:
                        # 训练时随机丢弃
                        if torch.rand(1) < self.survival_prob:
                            x = x + layer(x)
                    else:
                        # 推理时全部使用，乘以生存概率
                        x = x + self.survival_prob * layer(x)

                return x

        # 优势：
        # 1. 减少梯度消失
        # 2. 隐式集成（不同深度的子网络）
        # 3. 加速训练
```

### 4.2 感受野与分辨率

```python
class ReceptiveFieldAnalysis:
    """感受野分析"""

    def calculate_receptive_field(self):
        """
        计算感受野大小

        对于卷积层：
        RF_{l+1} = RF_l + (kernel_size - 1) * Π_{i=1}^l stride_i

        对于池化层：
        RF_{l+1} = RF_l * stride + (kernel_size - stride)
        """

        def compute_rf(layers_config):
            """
            layers_config: [(type, kernel_size, stride), ...]
            """
            rf = 1
            stride_product = 1

            for layer_type, k, s in layers_config:
                rf = rf + (k - 1) * stride_product
                stride_product *= s

            return rf

        # 示例：VGG-style网络
        vgg_config = [
            ('conv', 3, 1),
            ('conv', 3, 1),
            ('pool', 2, 2),
            ('conv', 3, 1),
            ('conv', 3, 1),
            ('pool', 2, 2),
            ('conv', 3, 1),
            ('conv', 3, 1),
            ('pool', 2, 2),
        ]

        rf_vgg = compute_rf(vgg_config)
        print(f"VGG感受野: {rf_vgg}")

        # 示例：ResNet-style网络
        resnet_config = [
            ('conv', 7, 2),  # stem
            ('pool', 3, 2),
            ('conv', 3, 1),
            ('conv', 3, 1),
            ('conv', 3, 2),
            ('conv', 3, 1),
        ]

        rf_resnet = compute_rf(resnet_config)
        print(f"ResNet感受野: {rf_resnet}")

    def design_for_target_receptive_field(self, target_rf=200):
        """
        为目标感受野设计网络
        """
        # 策略1：堆叠小卷积核（VGG风格）
        def vgg_style(target_rf):
            # 3x3卷积
            num_layers = (target_rf - 1) // 2 + 1
            return [('conv', 3, 1)] * num_layers

        # 策略2：使用大步幅（ResNet风格）
        def resnet_style(target_rf):
            # 结合步幅和卷积核
            return [
                ('conv', 7, 2),
                ('pool', 3, 2),
                ('conv', 3, 1),
                ('conv', 3, 2),
                ('conv', 3, 1),
            ]

        # 策略3：空洞卷积（DeepLab风格）
        def dilated_style(target_rf):
            # 指数增长的空洞率
            dilations = [1, 2, 4, 8, 16]
            return [('dilated_conv', 3, 1, d) for d in dilations]
```

---

## 总结

本教程深入讲解了神经网络架构设计的核心内容：

### 数学基础
- 通用逼近定理
- 激活函数性质
- 损失函数理论

### 核心组件
- 全连接层
- 卷积层
- 归一化层

### 现代架构
- 残差网络
- 注意力机制

### 设计原则
- 深度与宽度权衡
- 感受野设计

### 下一步
继续学习**教程四：优化器与训练技巧**
