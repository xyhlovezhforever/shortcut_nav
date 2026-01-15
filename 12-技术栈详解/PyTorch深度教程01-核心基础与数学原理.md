# PyTorch深度教程（一）：快速入门与核心概念

> **目标读者**：具有Python基础，追求十年经验级别的深度理解
> **学习路径**：从实践出发，在需要时补充数学原理

---

## 第一部分：5分钟快速上手

### 1.1 第一个神经网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 创建一个简单的线性回归模型
model = nn.Linear(in_features=1, out_features=1)

# 2. 准备数据（y = 2x + 3）
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[5.0], [7.0], [9.0], [11.0]])

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(100):
    # 前向传播
    y_pred = model(x_train)

    # 计算损失
    loss = criterion(y_pred, y_train)

    # 反向传播
    optimizer.zero_grad()  # 清空梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 5. 测试模型
with torch.no_grad():
    test_x = torch.tensor([[5.0]])
    test_y = model(test_x)
    print(f'Input: 5.0, Predicted: {test_y.item():.2f}, Expected: ~13.0')
```

**运行结果分析**：
- 模型学习到了线性关系 y ≈ 2x + 3
- 损失函数逐渐降低，说明模型在优化
- 最终预测结果接近真实值

---

## 第二部分：PyTorch核心架构

### 2.1 张量（Tensor）：PyTorch的核心数据结构

#### 2.1.1 什么是张量？

**简单理解**：
- 0维张量 = 标量（一个数字）：`torch.tensor(3.14)`
- 1维张量 = 向量（一串数字）：`torch.tensor([1, 2, 3])`
- 2维张量 = 矩阵（表格）：`torch.tensor([[1, 2], [3, 4]])`
- 3维张量 = 立方体（如RGB图像）：`torch.tensor([[[...]]])`

```python
# 创建张量的多种方式
import torch

# 1. 从Python列表创建
a = torch.tensor([1, 2, 3])
print(a)  # tensor([1, 2, 3])

# 2. 创建特殊张量
zeros = torch.zeros(2, 3)        # 全0矩阵
ones = torch.ones(2, 3)          # 全1矩阵
rand = torch.randn(2, 3)         # 标准正态分布随机数
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 3. 查看张量属性
print(f"形状: {rand.shape}")      # torch.Size([2, 3])
print(f"数据类型: {rand.dtype}")  # torch.float32
print(f"设备: {rand.device}")     # cpu 或 cuda:0
```

#### 2.1.2 张量的基本运算

```python
# 基础运算
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 逐元素运算
z1 = x + y      # 加法：[[6, 8], [10, 12]]
z2 = x * y      # 逐元素乘法：[[5, 12], [21, 32]]
z3 = x / y      # 除法
z4 = x ** 2     # 平方

# 矩阵运算
z5 = x @ y      # 矩阵乘法
z6 = x.T        # 转置

# 聚合运算
print(x.sum())      # 所有元素求和：10.0
print(x.mean())     # 平均值：2.5
print(x.max())      # 最大值：4.0
print(x.argmax())   # 最大值索引：3
```

<details>
<summary>📐 数学补充：矩阵乘法的数学原理（点击展开）</summary>

**💡 记忆口诀**：
```
行乘列，对应乘，求和填新位
左行数，右列数，中间要相等
```

**矩阵乘法定义**：
```
给定 A(m×n) 和 B(n×p)，结果 C(m×p) 的计算：
C[i,j] = Σ(k=1 to n) A[i,k] * B[k,j]

意思：左边第i行，乘以右边第j列，对应位置相乘再求和
```

**手工计算示例**：
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

AB = [(1×5 + 2×7)  (1×6 + 2×8)]   [19  22]
     [(3×5 + 4×7)  (3×6 + 4×8)] = [43  50]

第1行第1列: A的第1行 × B的第1列 = 1×5 + 2×7 = 19
第1行第2列: A的第1行 × B的第2列 = 1×6 + 2×8 = 22
```

**PyTorch验证**：
```python
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
print(A @ B)  # tensor([[19., 22.], [43., 50.]])
```

**几何意义**：
- 矩阵乘法 = 线性变换的组合
- A @ x：用矩阵A变换向量x
- 神经网络的每一层本质上就是矩阵乘法 + 非线性激活

</details>

#### 2.1.3 内存布局与存储机制

```python
class TensorStorageSystem:
    """
    深入理解PyTorch张量的存储系统
    """

    def storage_and_view(self):
        """
        存储（Storage）与视图（View）机制
        """
        # 1. 存储是一维连续内存块
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        print(f"存储内容: {x.storage()}")  # [1, 2, 3, 4, 5, 6]

        # 2. 张量是存储的视图
        # 通过offset, size, stride定义
        print(f"偏移量: {x.storage_offset()}")
        print(f"形状: {x.size()}")
        print(f"步幅: {x.stride()}")  # (3, 1) 表示行间隔3，列间隔1

        # 3. 视图操作零拷贝
        y = x.view(3, 2)  # 只改变形状元数据
        z = x.t()          # 转置只改变步幅

        # 4. 步幅的数学意义
        # 元素x[i,j]的内存地址：
        # addr = storage_offset + i*stride[0] + j*stride[1]

    def advanced_indexing(self):
        """
        高级索引与内存连续性
        """
        x = torch.randn(10, 20, 30)

        # 基础索引（不复制）
        y1 = x[0]         # shape: (20, 30)
        y2 = x[:, 0, :]   # shape: (10, 30)

        # 高级索引（复制数据）
        indices = torch.tensor([0, 2, 4])
        y3 = x[indices]   # 需要复制，因为索引不连续

        # 掩码索引
        mask = x > 0
        y4 = x[mask]      # 返回一维张量

    def memory_format(self):
        """
        内存格式：NCHW vs NHWC
        """
        # Channels Last格式（对某些硬件更优）
        x = torch.randn(8, 3, 224, 224)  # NCHW
        x_cl = x.to(memory_format=torch.channels_last)  # NHWC

        # 检查内存格式
        print(f"是否channels_last: {x_cl.is_contiguous(memory_format=torch.channels_last)}")

# 实战演示
demo = TensorStorageSystem()
demo.storage_and_view()
```

**为什么理解内存布局很重要？**
1. **性能优化**：连续内存访问更快
2. **避免意外bug**：理解何时发生数据复制
3. **GPU优化**：不同内存格式对GPU性能影响巨大

---

### 2.2 自动微分（Autograd）：反向传播的魔法

#### 2.2.1 梯度计算入门

```python
# 开启梯度追踪
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

# 计算梯度
y.backward()

print(f"x = {x.item()}")           # 2.0
print(f"y = {y.item()}")           # 4 + 6 + 1 = 11
print(f"dy/dx = {x.grad.item()}")  # 2*2 + 3 = 7
```

<details>
<summary>📐 数学补充：梯度的含义（点击展开）</summary>

**💡 记忆口诀**：
```
梯度就是斜率坡，告诉方向怎么走
自变量变一小步，因变量跟着走
```

**梯度定义**：
```
对于函数 y = f(x)，梯度 dy/dx 表示：
- x变化一个小量Δx时，y变化多少
- 函数在该点的斜率
- 优化时的前进方向（负梯度 = 下降最快）
```

**例子**：y = x² + 3x + 1
```
dy/dx = 2x + 3

当 x = 2：
dy/dx = 2(2) + 3 = 7

含义：x从2增加到2.01时，
     y大约增加 7 × 0.01 = 0.07
```

**验证**：
```python
x = 2.0
y1 = x**2 + 3*x + 1  # 11.0
x = 2.01
y2 = x**2 + 3*x + 1  # 11.0701
print(f"实际变化: {y2 - y1:.4f}")      # 0.0701
print(f"梯度预测: {7 * 0.01:.4f}")     # 0.0700
```

</details>

#### 2.2.2 多变量梯度

```python
# 多变量函数
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0]**2 + 3*x[0]*x[1] + x[1]**2

y.backward()

print(f"∂y/∂x₀ = {x.grad[0].item()}")  # 2*1 + 3*2 = 8
print(f"∂y/∂x₁ = {x.grad[1].item()}")  # 3*1 + 2*2 = 7
```

#### 2.2.3 计算图与反向传播

```python
class ComputationGraphDemo:
    """
    理解PyTorch的动态计算图
    """

    def visualize_graph(self):
        """
        可视化计算图
        """
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)

        # 构建计算图
        a = x + y       # a = 5.0
        b = x * y       # b = 6.0
        c = a * b       # c = 30.0

        c.backward()

        print(f"dc/dx = {x.grad}")  # 11.0
        print(f"dc/dy = {y.grad}")  # 13.0

        """
        计算图：

        x=2 ─┬─(+)─→ a=5 ─┐
             │            ├─(*)─→ c=30
        y=3 ─┼─(*)─→ b=6 ─┘
             │

        反向传播：
        dc/dx = dc/da * da/dx + dc/db * db/dx
              = b * 1 + a * y
              = 6 + 5*1 = 11
        """

    def gradient_accumulation(self):
        """
        梯度累积机制
        """
        x = torch.tensor(1.0, requires_grad=True)

        # 第一次前向+反向
        y1 = x ** 2
        y1.backward()
        print(f"第一次梯度: {x.grad}")  # 2.0

        # 不清零，再次反向（梯度会累积！）
        y2 = x ** 3
        y2.backward()
        print(f"累积后梯度: {x.grad}")  # 2.0 + 3.0 = 5.0

        # 这就是为什么训练时需要 optimizer.zero_grad()

# 实战演示
demo = ComputationGraphDemo()
demo.visualize_graph()
demo.gradient_accumulation()
```

<details>
<summary>📐 数学补充：链式法则与反向传播（点击展开）</summary>

**💡 记忆口诀**：
```
链式求导像接力，一棒一棒往回递
外层导数乘内层，从后往前全连起
```

**链式法则（Chain Rule）**：
```
如果 z = f(y), y = g(x)，那么：
dz/dx = (dz/dy) × (dy/dx)

意思：z对x的导数 = z对y的导数 × y对x的导数
就像接力赛，梯度从后往前一层层传
```

**反向传播本质**：
- 就是链式法则的递归应用
- 从输出节点开始，逐层向后计算梯度
- 神经网络训练的核心算法

**例子**：z = (x² + 1)³
```
设 y = x² + 1, z = y³

前向传播（从左到右算值）：
x = 2 → y = 5 → z = 125

反向传播（从右到左算梯度）：
dz/dy = 3y² = 75
dy/dx = 2x = 4
dz/dx = (dz/dy) × (dy/dx) = 75 × 4 = 300
```

**PyTorch验证**：
```python
x = torch.tensor(2.0, requires_grad=True)
z = (x**2 + 1)**3
z.backward()
print(x.grad)  # tensor(300.)
```

</details>

---

### 2.3 神经网络模块（nn.Module）

#### 2.3.1 构建自定义网络

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播逻辑
        x = F.relu(self.fc1(x))  # 隐藏层 + ReLU激活
        x = self.fc2(x)           # 输出层
        return x

# 创建模型
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)

# 查看模型结构
print(model)

# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

**输出解释**：
```
SimpleNN(
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (fc2): Linear(in_features=20, out_features=2, bias=True)
)

fc1.weight: torch.Size([20, 10])  # 第一层权重矩阵
fc1.bias: torch.Size([20])         # 第一层偏置
fc2.weight: torch.Size([2, 20])    # 第二层权重矩阵
fc2.bias: torch.Size([2])          # 第二层偏置
```

#### 2.3.2 激活函数

```python
# 常用激活函数
x = torch.linspace(-5, 5, 100)

# ReLU: max(0, x)
relu_out = F.relu(x)

# Sigmoid: 1 / (1 + e^(-x))
sigmoid_out = torch.sigmoid(x)

# Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
tanh_out = torch.tanh(x)

# LeakyReLU: max(0.01x, x)
leaky_relu_out = F.leaky_relu(x, 0.01)

# GELU (现代Transformer常用)
gelu_out = F.gelu(x)
```

**为什么需要激活函数？**
- 没有激活函数，多层神经网络 = 一个线性变换
- 激活函数引入非线性，让网络能拟合复杂函数

<details>
<summary>📐 数学补充：为什么线性层叠加还是线性？（点击展开）</summary>

**💡 记忆口诀**：
```
线性套线性，还是线性变
加个激活函数，才能拐弯弯
```

**证明**：
```
假设两层线性变换：
h = W₁x + b₁
y = W₂h + b₂

代入：
y = W₂(W₁x + b₁) + b₂
  = W₂W₁x + W₂b₁ + b₂
  = Wx + b  （其中 W = W₂W₁, b = W₂b₁ + b₂）

结论：两层线性变换等价于一层！
就像 y = 2(3x + 1) + 5 = 6x + 7，最终还是一条直线
```

**加入激活函数后**：
```
h = σ(W₁x + b₁)   # σ是激活函数（如ReLU）
y = W₂h + b₂

此时无法简化为单层，因为σ是非线性的
就像先折叠再拉伸，能拟合复杂曲线
```

**目的**：
- 没有激活函数：多层网络 = 浪费计算
- 有激活函数：能学习复杂的非线性模式

</details>

---

### 2.4 设备管理（CPU vs GPU）

```python
# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将张量移到GPU
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to(device)

# 将模型移到GPU
model = SimpleNN(10, 20, 2)
model = model.to(device)

# 完整训练循环示例
def train_on_gpu(model, data_loader, optimizer, criterion, device):
    model.train()
    for batch_x, batch_y in data_loader:
        # 数据移到GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**性能对比**：
```python
import time

# CPU版本
x = torch.randn(5000, 5000)
y = torch.randn(5000, 5000)

start = time.time()
z = x @ y
print(f"CPU耗时: {time.time() - start:.4f}秒")

# GPU版本（如果可用）
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    y_gpu = y.to('cuda')

    start = time.time()
    z_gpu = x_gpu @ y_gpu
    torch.cuda.synchronize()  # 等待GPU计算完成
    print(f"GPU耗时: {time.time() - start:.4f}秒")
```

---

## 第三部分：数值稳定性与实践技巧

### 3.1 数值稳定性问题

#### 3.1.1 指数函数的数值稳定实现

```python
def softmax_naive(x):
    """
    朴素实现（数值不稳定）
    """
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def softmax_stable(x):
    """
    数值稳定版本
    """
    # 减去最大值，防止指数溢出
    x_max = x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# 测试
x = torch.tensor([1000., 1001., 1002.])

try:
    result1 = softmax_naive(x)
    print(f"朴素版本: {result1}")  # 可能出现NaN
except:
    print("朴素版本溢出！")

result2 = softmax_stable(x)
print(f"稳定版本: {result2}")  # tensor([0.0900, 0.2447, 0.6652])
```

<details>
<summary>📐 数学补充：Softmax数值稳定性（点击展开）</summary>

**💡 记忆口诀**：
```
指数函数易爆炸，减去最大保平安
上下同除不改值，稳定计算是关键
```

**Softmax定义**：
```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

作用：把一组数字转换成概率（和为1，都在0-1之间）
```

**问题**：当x很大时，exp(x)会溢出
```
exp(1000) = ∞ （超出浮点数范围）
```

**解决方案**：利用数学恒等式
```
softmax(x) = softmax(x - c)  对任意常数c成立

证明：
softmax(xᵢ - c) = exp(xᵢ - c) / Σⱼ exp(xⱼ - c)
                = [exp(xᵢ) / exp(c)] / [Σⱼ exp(xⱼ) / exp(c)]
                = exp(xᵢ) / Σⱼ exp(xⱼ)
                = softmax(xᵢ)

选择 c = max(x)，可以保证：
- 所有 exp(xᵢ - c) ≤ 1（防止溢出）
- 至少一个 exp(xᵢ - c) = 1（防止下溢）
```

**目的**：
- 朴素实现：exp(大数) → 溢出 → NaN
- 稳定实现：减最大值 → 数值在安全范围 → 正确结果

</details>

#### 3.1.2 梯度消失与爆炸

```python
class GradientFlowDemo:
    """
    演示梯度消失和梯度爆炸
    """

    def gradient_vanishing(self):
        """
        梯度消失示例
        """
        # 使用Sigmoid激活函数的深层网络
        x = torch.randn(1, 10, requires_grad=True)

        # 10层网络，每层都用Sigmoid
        h = x
        for _ in range(10):
            W = torch.randn(10, 10) * 0.5
            h = torch.sigmoid(h @ W)

        loss = h.sum()
        loss.backward()

        print(f"输入梯度范数: {x.grad.norm().item():.6f}")
        # 很小的值，说明梯度消失了

    def gradient_exploding(self):
        """
        梯度爆炸示例
        """
        x = torch.randn(1, 10, requires_grad=True)

        # 权重过大导致梯度爆炸
        h = x
        for _ in range(10):
            W = torch.randn(10, 10) * 2  # 权重较大
            h = h @ W

        loss = h.sum()
        loss.backward()

        print(f"输入梯度范数: {x.grad.norm().item():.6f}")
        # 很大的值，说明梯度爆炸了

demo = GradientFlowDemo()
demo.gradient_vanishing()
demo.gradient_exploding()
```

**解决方案**：
1. **批归一化（Batch Normalization）**
2. **残差连接（ResNet）**
3. **合适的权重初始化**
4. **梯度裁剪**

---

### 3.2 权重初始化策略

```python
import math

def xavier_uniform_init(tensor, gain=1.0):
    """
    Xavier均匀初始化（适用于Tanh、Sigmoid）
    """
    fan_in, fan_out = tensor.shape
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-a, a)

def kaiming_normal_init(tensor, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming正态初始化（适用于ReLU）
    """
    fan_in, fan_out = tensor.shape
    fan = fan_in if mode == 'fan_in' else fan_out

    # ReLU的增益因子
    gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / math.sqrt(fan)

    with torch.no_grad():
        tensor.normal_(0, std)

# PyTorch内置初始化
layer = nn.Linear(100, 50)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# 或者
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

<details>
<summary>📐 数学补充：Xavier初始化的数学原理（点击展开）</summary>

**💡 记忆口诀**：
```
输入输出取平均，方差倒数是关键
前向反向都稳定，信号传播不衰减
```

**目标**：保持信号的方差在前向和反向传播中不变

**推导**（简化版）：
```
假设输入 x 的方差为 Var(x) = 1
层输出 y = Wx（忽略偏置）

Var(y) = Var(Σᵢ wᵢxᵢ)
       = Σᵢ Var(wᵢxᵢ)  （假设独立）
       = Σᵢ E[wᵢ²]E[xᵢ²]  （假设零均值）
       = n_in × Var(w) × Var(x)

要使 Var(y) = Var(x)，需要：
n_in × Var(w) = 1
Var(w) = 1 / n_in

同理，考虑反向传播，得到：
Var(w) = 1 / n_out

折中：Var(w) = 2 / (n_in + n_out)

对于均匀分布 U(-a, a)：
Var = a² / 3
所以 a = sqrt(6 / (n_in + n_out))
```

**目的**：
- 太小：信号逐层衰减，梯度消失
- 太大：信号逐层放大，梯度爆炸
- 刚好：信号稳定传播，训练顺畅

**适用场景**：
- Xavier：适合Tanh、Sigmoid等对称激活函数
- Kaiming：适合ReLU（会砍掉一半，需要更大的初始化）

</details>

---

## 第四部分：完整实战案例

### 4.1 图像分类：MNIST手写数字识别

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 2. 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# 3. 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f' Loss: {loss.item():.6f}')

# 4. 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 5. 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 6. 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

---

## 总结与学习路线图

### 你已经学会了什么

✅ **核心概念**：
- 张量的创建、操作和内存管理
- 自动微分与梯度计算
- 神经网络的构建与训练
- GPU加速

✅ **实践技能**：
- 数值稳定性的处理
- 权重初始化策略
- 完整的训练流程
- 模型的保存与加载

### 下一步学习方向

📖 **教程02 - 张量运算与自动微分深度解析**：
- 高级索引与广播机制
- 自定义autograd函数
- 内存优化技巧

📖 **教程03 - 神经网络架构设计**：
- CNN、RNN、Transformer详解
- 自定义网络层
- 模型组合技巧

📖 **教程04 - 优化器与训练技巧**：
- 各种优化器的数学原理
- 学习率调度
- 正则化技术

📖 **教程05 - 分布式训练与性能优化**：
- 数据并行与模型并行
- 混合精度训练
- 性能分析工具

📖 **教程06 - 实战应用案例**：
- 计算机视觉
- 自然语言处理
- 生成模型

---

## 附录：常用数学知识速查

<details>
<summary>📐 线性代数基础（点击展开）</summary>

### 向量运算
```python
# 向量点积
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
dot_product = (a * b).sum()  # 32.0

# 向量范数
l1_norm = a.abs().sum()       # L1范数: 6.0
l2_norm = a.norm()            # L2范数: 3.742
```

### 矩阵运算
```python
# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # (3, 5)

# 矩阵转置
A_T = A.T

# 逆矩阵
A_square = torch.randn(3, 3)
A_inv = torch.inverse(A_square)
```

### 特征值分解
```python
A = torch.randn(3, 3)
eigenvalues, eigenvectors = torch.linalg.eig(A)
```

</details>

<details>
<summary>📐 微积分基础（点击展开）</summary>

### 常用导数
```
d/dx (x^n) = nx^(n-1)
d/dx (e^x) = e^x
d/dx (ln x) = 1/x
d/dx (sin x) = cos x
d/dx (cos x) = -sin x
```

### 链式法则
```
d/dx f(g(x)) = f'(g(x)) × g'(x)
```

### 常用激活函数的导数
```python
# Sigmoid: σ(x) = 1/(1+e^(-x))
# 导数: σ'(x) = σ(x)(1-σ(x))

# ReLU: f(x) = max(0, x)
# 导数: f'(x) = 1 if x>0 else 0

# Tanh: tanh(x)
# 导数: 1 - tanh²(x)
```

</details>

<details>
<summary>📐 概率论基础（点击展开）</summary>

### 常用分布
```python
# 正态分布
x = torch.randn(1000)  # N(0, 1)

# 均匀分布
x = torch.rand(1000)   # U(0, 1)

# 伯努利分布
x = torch.bernoulli(torch.full((1000,), 0.5))
```

### 期望与方差
```python
# 期望（均值）
mean = x.mean()

# 方差
var = x.var()

# 标准差
std = x.std()
```

</details>

---

**恭喜你完成第一章的学习！** 🎉

现在你已经掌握了PyTorch的核心基础，可以开始构建自己的深度学习模型了。记住：
- 多动手实践，理论结合代码
- 遇到数学概念时，查看补充说明
- 循序渐进，不要急于求成

继续加油！💪
