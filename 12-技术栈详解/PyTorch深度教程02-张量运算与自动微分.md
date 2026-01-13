# PyTorch深度教程（二）：张量运算与自动微分深度解析

> **前置要求**：完成教程一的学习
> **核心目标**：掌握张量运算的数学本质与自动微分的工程实现

---

## 第一部分：张量运算的数学与实现

### 1.1 张量的数学结构

#### 1.1.1 张量的严格定义

```python
"""
数学定义：
(p, q)型张量T是一个多重线性映射：
T: V* × ... × V* (p个) × V × ... × V (q个) → ℝ

其中：
- V是向量空间
- V*是对偶空间
- p是逆变指标数（上标）
- q是协变指标数（下标）
"""

import torch
import numpy as np

class TensorMathematics:
    """张量数学理论与实现"""

    def tensor_transformations(self):
        """
        张量在坐标变换下的行为
        """
        # 向量（1,0张量）：逆变
        # v'ⁱ = ∂x'ⁱ/∂xʲ vʲ

        # 对偶向量（0,1张量）：协变
        # ω'ᵢ = ∂xʲ/∂x'ⁱ ωⱼ

        # 示例：2D旋转变换
        theta = np.pi / 4  # 45度
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=torch.float32)

        # 向量变换
        v = torch.tensor([1., 0.])
        v_prime = rotation_matrix @ v

        # 对偶向量变换
        omega = torch.tensor([1., 0.])
        omega_prime = omega @ rotation_matrix.t()

        print(f"变换后的向量: {v_prime}")
        print(f"变换后的对偶向量: {omega_prime}")

    def tensor_products(self):
        """
        张量积（Tensor Product）
        """
        # 1. 外积（Outer Product）
        # u ⊗ v 产生一个矩阵
        u = torch.tensor([1., 2., 3.])
        v = torch.tensor([4., 5.])
        outer = torch.outer(u, v)  # shape: (3, 2)
        print(f"外积:\n{outer}")

        # 2. Kronecker积
        # A ⊗ B 的每个元素 aᵢⱼB
        A = torch.tensor([[1., 2.], [3., 4.]])
        B = torch.tensor([[5., 6.], [7., 8.]])
        kron = torch.kron(A, B)
        print(f"Kronecker积:\n{kron}")

        # 3. 张量缩并（Tensor Contraction）
        # 爱因斯坦求和约定
        # Cᵢₖ = Σⱼ AᵢⱼBⱼₖ
        C = torch.einsum('ij,jk->ik', A, B)
        print(f"矩阵乘法（缩并）:\n{C}")
```

#### 1.1.2 Einstein求和约定

```python
class EinsteinSummation:
    """
    爱因斯坦求和约定：张量运算的通用语言
    """

    def einsum_basics(self):
        """
        einsum基础语法
        """
        # 基本规则：
        # 1. 重复索引表示求和
        # 2. 未重复索引出现在输出中

        # 示例1：矩阵乘法
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = torch.einsum('ik,kj->ij', A, B)  # 等价于A @ B

        # 示例2：批量矩阵乘法
        A = torch.randn(10, 3, 4)  # batch矩阵
        B = torch.randn(10, 4, 5)
        C = torch.einsum('bik,bkj->bij', A, B)  # 等价于torch.bmm

        # 示例3：迹（Trace）
        A = torch.randn(5, 5)
        trace = torch.einsum('ii->', A)  # Σ Aᵢᵢ

        # 示例4：对角元素
        diag = torch.einsum('ii->i', A)  # [A₀₀, A₁₁, ...]

    def advanced_einsum_operations(self):
        """
        高级einsum运算
        """
        # 1. 双线性形式：xᵀAy
        A = torch.randn(5, 5)
        x = torch.randn(5)
        y = torch.randn(5)
        result = torch.einsum('i,ij,j->', x, A, y)

        # 2. 注意力机制中的运算
        # Attention(Q,K,V) = softmax(QKᵀ/√d)V
        Q = torch.randn(10, 8, 64)  # (batch, seq, dim)
        K = torch.randn(10, 8, 64)
        V = torch.randn(10, 8, 64)

        # QKᵀ: (batch, seq_q, seq_k)
        scores = torch.einsum('bqd,bkd->bqk', Q, K)

        # softmax(scores)V: (batch, seq_q, dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum('bqk,bkd->bqd', attn_weights, V)

        # 3. 图卷积网络
        # H' = σ(ÂHW)
        # Â: 归一化邻接矩阵
        A_norm = torch.randn(100, 100)  # 节点数100
        H = torch.randn(100, 64)        # 特征维度64
        W = torch.randn(64, 128)        # 输出维度128

        H_prime = torch.einsum('ij,jd,dk->ik', A_norm, H, W)

    def einsum_optimization(self):
        """
        einsum的性能优化
        """
        # opt_einsum库可以优化复杂的缩并路径
        # 例如：A_ij B_jk C_kl D_lm
        # 不同的计算顺序有不同的复杂度

        A = torch.randn(10, 20)
        B = torch.randn(20, 30)
        C = torch.randn(30, 40)
        D = torch.randn(40, 50)

        # PyTorch会自动优化
        result = torch.einsum('ij,jk,kl,lm->im', A, B, C, D)

        # 手动指定路径（高级用法）
        # result = torch.einsum('ij,jk,kl,lm->im', A, B, C, D,
        #                       optimize='optimal')
```

### 1.2 高性能张量运算

#### 1.2.1 内存高效的张量操作

```python
class MemoryEfficientOperations:
    """
    内存高效的张量运算技巧
    """

    def inplace_operations(self):
        """
        原地操作（In-place Operations）
        """
        # 1. 原地操作的标记：尾随下划线
        x = torch.randn(1000, 1000)

        # 非原地：创建新张量
        y = x.add(1.0)  # 需要额外内存

        # 原地：修改现有张量
        x.add_(1.0)  # 节省内存

        # 2. 原地操作的限制
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x ** 2

        # 错误：会破坏计算图
        # x.add_(1.0)  # RuntimeError

        # 3. 安全的原地操作
        with torch.no_grad():
            x.add_(1.0)  # OK，不需要梯度

    def view_vs_reshape(self):
        """
        view vs reshape：什么时候复制数据？
        """
        x = torch.randn(4, 4)

        # view：要求张量连续，不复制数据
        y = x.view(16)  # OK，共享存储

        # 转置破坏连续性
        z = x.t()  # 步幅改变
        # w = z.view(16)  # 错误：不连续

        # 解决方案1：contiguous()
        w = z.contiguous().view(16)  # 复制数据

        # 解决方案2：reshape
        w = z.reshape(16)  # 自动处理，必要时复制

        # 检查是否复制
        print(f"x和y共享存储: {x.data_ptr() == y.data_ptr()}")
        print(f"z是否连续: {z.is_contiguous()}")

    def memory_layout_optimization(self):
        """
        内存布局优化
        """
        # 1. Channels Last格式
        # NCHW (默认) vs NHWC (channels_last)
        img = torch.randn(4, 3, 224, 224)  # NCHW

        # 转换为channels_last
        img_cl = img.to(memory_format=torch.channels_last)

        # 好处：某些操作（如卷积）更快
        conv = torch.nn.Conv2d(3, 64, 3, padding=1)
        conv = conv.to(memory_format=torch.channels_last)

        # 2. 步幅分析
        print(f"NCHW步幅: {img.stride()}")      # (150528, 50176, 224, 1)
        print(f"NHWC步幅: {img_cl.stride()}")   # (150528, 1, 672, 3)

    def broadcasting_efficient(self):
        """
        高效的广播操作
        """
        # 广播规则：
        # 1. 对齐右侧维度
        # 2. 大小为1的维度可以广播

        # 示例：批量归一化
        x = torch.randn(100, 64, 28, 28)  # (N, C, H, W)
        mean = torch.randn(64)             # (C,)
        std = torch.randn(64)

        # 方法1：低效（创建中间张量）
        mean_expanded = mean.view(1, 64, 1, 1).expand_as(x)
        x_norm = (x - mean_expanded) / std.view(1, 64, 1, 1)

        # 方法2：高效（直接广播）
        x_norm = (x - mean.view(1, 64, 1, 1)) / std.view(1, 64, 1, 1)

        # 方法3：最高效（使用unsqueeze）
        x_norm = (x - mean[None, :, None, None]) / std[None, :, None, None]
```

#### 1.2.2 高级索引与切片

```python
class AdvancedIndexing:
    """
    高级索引技术
    """

    def indexing_types(self):
        """
        索引类型及其性能
        """
        x = torch.randn(100, 100)

        # 1. 基础索引（视图，不复制）
        y = x[10:20, :]      # 行切片
        z = x[:, [0, 1, 2]]  # 列索引（复制！）

        # 2. 高级索引（复制数据）
        indices = torch.tensor([0, 2, 4, 6, 8])
        y = x[indices]  # 选择特定行

        # 3. 布尔索引（复制数据）
        mask = x > 0
        y = x[mask]  # 返回一维张量

        # 4. 组合索引
        # 选择特定位置的元素
        rows = torch.tensor([0, 1, 2])
        cols = torch.tensor([0, 1, 0])
        y = x[rows, cols]  # [x[0,0], x[1,1], x[2,0]]

    def gather_scatter_operations(self):
        """
        gather和scatter：高级索引的高效替代
        """
        # 1. gather：从源张量收集值
        src = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])

        # 每行收集不同索引
        indices = torch.tensor([[0, 0],
                                [1, 0],
                                [2, 1]])

        # 沿dim=1收集
        gathered = torch.gather(src, dim=1, index=indices)
        print(f"Gathered:\n{gathered}")
        # [[1, 1],
        #  [5, 4],
        #  [9, 8]]

        # 2. scatter：将值分散到目标张量
        dst = torch.zeros(3, 3)
        src_values = torch.tensor([[10., 20.],
                                    [30., 40.],
                                    [50., 60.]])

        dst.scatter_(dim=1, index=indices, src=src_values)
        print(f"Scattered:\n{dst}")

        # 3. 应用：Top-K采样
        def top_k_sampling(logits, k=5):
            """
            从logits中采样top-k个token
            """
            # 获取top-k值和索引
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

            # 对top-k值做softmax
            probs = torch.softmax(top_k_values, dim=-1)

            # 采样
            sampled_idx = torch.multinomial(probs, num_samples=1)

            # 映射回原始索引
            token_idx = torch.gather(top_k_indices, -1, sampled_idx)
            return token_idx

    def advanced_masking(self):
        """
        高级掩码技术
        """
        # 1. 注意力掩码
        seq_len = 10
        # 因果掩码（下三角）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        # 填充掩码
        lengths = torch.tensor([5, 7, 10])  # 序列长度
        batch_size = len(lengths)

        # 创建掩码：(batch, seq_len)
        mask = torch.arange(seq_len)[None, :] < lengths[:, None]

        # 2. masked_fill：高效替换值
        scores = torch.randn(3, 10, 10)
        scores.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        # 3. masked_select vs 布尔索引
        x = torch.randn(1000, 1000)
        mask = x > 0

        # 方法1：布尔索引
        positive1 = x[mask]

        # 方法2：masked_select
        positive2 = torch.masked_select(x, mask)

        # 性能相似，但masked_select更显式
```

---

## 第二部分：自动微分系统深度解析

### 2.1 计算图的内部实现

#### 2.1.1 动态计算图的数据结构

```python
class ComputationGraphInternals:
    """
    计算图的内部数据结构
    """

    def graph_node_structure(self):
        """
        图节点的结构
        """
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x ** 2
        z = y.sum()

        # 节点属性
        print(f"z.grad_fn: {z.grad_fn}")              # SumBackward0
        print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}")

        # 遍历计算图
        def print_graph(var, indent=0):
            """递归打印计算图"""
            print('  ' * indent + str(var))
            if hasattr(var, 'next_functions'):
                for fn, _ in var.next_functions:
                    if fn is not None:
                        print_graph(fn, indent + 1)

        print("\n计算图结构:")
        print_graph(z.grad_fn)

    def saved_tensors_mechanism(self):
        """
        saved_tensors：反向传播的关键
        """
        class DebugFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                # 保存反向传播需要的张量
                ctx.save_for_backward(x, weight)
                output = x @ weight
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # 获取保存的张量
                x, weight = ctx.saved_tensors

                # 计算梯度
                grad_x = grad_output @ weight.t()
                grad_weight = x.t() @ grad_output

                return grad_x, grad_weight

        # 使用
        x = torch.randn(10, 5, requires_grad=True)
        weight = torch.randn(5, 3, requires_grad=True)
        output = DebugFunction.apply(x, weight)

        # saved_tensors占用内存
        print(f"保存的张量: {output.grad_fn.saved_tensors}")

    def graph_lifecycle(self):
        """
        计算图的生命周期
        """
        # 1. 图构建
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x ** 2
        z = y.sum()

        # 2. 反向传播
        z.backward()  # 计算梯度

        # 3. 图释放
        # 默认情况下，backward后图被释放
        # z.backward()  # 错误！图已释放

        # 保留图
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x ** 2
        z = y.sum()
        z.backward(retain_graph=True)  # 保留图
        z.backward()  # OK

        # 4. 手动分离
        y_detached = y.detach()  # 切断梯度流
        # y_detached.backward()  # 不会计算x的梯度
```

#### 2.1.2 梯度累积机制

```python
class GradientAccumulation:
    """
    梯度累积的数学与实现
    """

    def basic_accumulation(self):
        """
        基础梯度累积
        """
        x = torch.tensor([1., 2., 3.], requires_grad=True)

        # 第一次计算
        y = x ** 2
        z1 = y.sum()
        z1.backward(retain_graph=True)
        print(f"第一次梯度: {x.grad}")  # [2, 4, 6]

        # 第二次计算（梯度累积！）
        z2 = (x ** 3).sum()
        z2.backward()
        print(f"累积后梯度: {x.grad}")  # [5, 16, 33] = [2,4,6] + [3,12,27]

        # 清零梯度
        x.grad.zero_()

    def gradient_accumulation_for_large_batches(self):
        """
        用梯度累积模拟大批量训练
        """
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 模拟大批量（batch_size=64）
        # 实际内存只能容纳16
        accumulation_steps = 4
        effective_batch_size = 16 * accumulation_steps

        for step in range(accumulation_steps):
            # 小批量数据
            x = torch.randn(16, 10)
            y = torch.randn(16, 1)

            # 前向传播
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)

            # 反向传播（梯度累积）
            loss = loss / accumulation_steps  # 关键：归一化
            loss.backward()

        # 统一更新
        optimizer.step()
        optimizer.zero_grad()

    def selective_gradient_computation(self):
        """
        选择性梯度计算
        """
        # 1. 冻结部分参数
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )

        # 冻结第一层
        for param in model[0].parameters():
            param.requires_grad = False

        # 只有第二层会计算梯度

        # 2. 梯度检查点（checkpoint）
        from torch.utils.checkpoint import checkpoint

        def expensive_function(x):
            return x ** 2 + torch.sin(x)

        x = torch.randn(100, 100, requires_grad=True)

        # 不保存中间激活
        y = checkpoint(expensive_function, x)

        # 反向传播时重新计算
```

### 2.2 自定义autograd函数

#### 2.2.1 高级Function实现

```python
class AdvancedAutogradFunctions:
    """
    高级自定义autograd函数
    """

    @staticmethod
    def custom_relu_with_stats():
        """
        带统计信息的自定义ReLU
        """
        class ReLUWithStats(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 保存统计信息
                mask = x > 0
                ctx.save_for_backward(mask)

                # 统计激活神经元比例
                ctx.activation_ratio = mask.float().mean().item()

                return x * mask

            @staticmethod
            def backward(ctx, grad_output):
                mask, = ctx.saved_tensors
                grad_input = grad_output * mask
                return grad_input

            @staticmethod
            def get_activation_ratio(ctx):
                return ctx.activation_ratio

        return ReLUWithStats

    @staticmethod
    def numerical_stable_softmax():
        """
        数值稳定的Softmax实现
        """
        class StableSoftmax(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 数值稳定技巧
                x_max = x.max(dim=-1, keepdim=True)[0]
                exp_x = torch.exp(x - x_max)
                softmax_x = exp_x / exp_x.sum(dim=-1, keepdim=True)

                ctx.save_for_backward(softmax_x)
                return softmax_x

            @staticmethod
            def backward(ctx, grad_output):
                """
                Softmax的Jacobian矩阵：
                ∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ - sⱼ)

                向量形式：
                ∂s/∂x = diag(s) - s⊗s
                """
                softmax_x, = ctx.saved_tensors

                # 计算 grad_output · Jacobian
                grad_input = softmax_x * grad_output
                sum_grad = grad_input.sum(dim=-1, keepdim=True)
                grad_input -= softmax_x * sum_grad

                return grad_input

        return StableSoftmax

    @staticmethod
    def sparse_operations():
        """
        稀疏张量的自定义操作
        """
        class SparseLinear(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight):
                """
                input: dense (N, in_features)
                weight: sparse (out_features, in_features)
                """
                ctx.save_for_backward(input, weight)
                # 稀疏矩阵乘法
                output = torch.sparse.mm(weight, input.t()).t()
                return output

            @staticmethod
            def backward(ctx, grad_output):
                input, weight = ctx.saved_tensors

                # 输入的梯度
                grad_input = torch.sparse.mm(
                    weight.t(),
                    grad_output.t()
                ).t()

                # 权重的梯度（保持稀疏）
                grad_weight = torch.sparse.mm(
                    grad_output.t(),
                    input
                )

                return grad_input, grad_weight

        return SparseLinear
```

#### 2.2.2 双向传播与高阶导数

```python
class HigherOrderGradients:
    """
    高阶导数计算
    """

    def second_order_gradients(self):
        """
        二阶导数：Hessian矩阵
        """
        def compute_hessian_vector_product(f, x, v):
            """
            计算Hessian-向量积：H·v
            不显式构造Hessian矩阵
            """
            # 一阶导数
            x.requires_grad_(True)
            y = f(x)
            grad_y = torch.autograd.grad(y, x, create_graph=True)[0]

            # Hessian-向量积
            hvp = torch.autograd.grad(grad_y, x, v, retain_graph=True)[0]
            return hvp

        # 示例：f(x) = x^T A x
        A = torch.randn(5, 5)
        A = A + A.t()  # 对称矩阵

        def f(x):
            return 0.5 * x @ A @ x

        x = torch.randn(5)
        v = torch.randn(5)

        hvp = compute_hessian_vector_product(f, x, v)
        # 理论值：H·v = A·v
        print(f"Hessian-向量积: {hvp}")
        print(f"理论值 (A·v): {A @ v}")

    def jacobian_computation(self):
        """
        雅可比矩阵的高效计算
        """
        def compute_jacobian(func, x, vectorize=True):
            """
            计算Jacobian矩阵
            vectorize=True: 使用vmap加速
            """
            x = x.detach().requires_grad_(True)
            y = func(x)

            if vectorize and hasattr(torch, 'vmap'):
                # 使用vmap向量化（PyTorch 2.0+）
                def get_vjp(v):
                    return torch.autograd.grad(y, x, v, retain_graph=True)[0]

                eye = torch.eye(y.numel(), device=y.device)
                jacobian = torch.vmap(get_vjp)(eye)
            else:
                # 传统方法
                jacobian = []
                for i in range(y.numel()):
                    grad_outputs = torch.zeros_like(y)
                    grad_outputs.view(-1)[i] = 1
                    grads = torch.autograd.grad(
                        y, x, grad_outputs, retain_graph=True
                    )[0]
                    jacobian.append(grads.view(-1))
                jacobian = torch.stack(jacobian)

            return jacobian

        # 示例
        def f(x):
            return torch.stack([x[0]**2 + x[1], x[0] * x[1]])

        x = torch.tensor([2., 3.], requires_grad=True)
        J = compute_jacobian(f, x, vectorize=False)
        print(f"Jacobian矩阵:\n{J}")

    def forward_mode_ad(self):
        """
        前向模式自动微分
        用于Jacobian宽度小的情况
        """
        # PyTorch的前向模式AD（实验性）
        try:
            from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

            x = torch.tensor([1., 2., 3.])

            with dual_level():
                # 创建对偶数: x + ε·v
                v = torch.tensor([1., 0., 0.])  # 方向导数的方向
                x_dual = make_dual(x, v)

                # 计算
                y_dual = torch.sin(x_dual) + x_dual ** 2

                # 提取原始值和导数
                y, dy_dx_v = unpack_dual(y_dual)
                print(f"方向导数: {dy_dx_v}")

        except ImportError:
            print("前向模式AD需要PyTorch 1.11+")
```

### 2.3 梯度优化技术

#### 2.3.1 梯度裁剪策略

```python
class GradientClippingStrategies:
    """
    梯度裁剪的各种策略
    """

    def norm_based_clipping(self, parameters, max_norm, norm_type=2):
        """
        基于范数的梯度裁剪
        """
        # 计算总梯度范数
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        if norm_type == float('inf'):
            # L∞范数
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            # Lp范数
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
                norm_type
            )

        # 裁剪系数
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

        return total_norm

    def value_based_clipping(self, parameters, clip_value):
        """
        基于值的梯度裁剪
        """
        for p in parameters:
            if p.grad is not None:
                p.grad.data.clamp_(-clip_value, clip_value)

    def adaptive_clipping(self, parameters, percentile=95):
        """
        自适应梯度裁剪
        基于梯度分布的百分位数
        """
        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.data.abs().view(-1))

        all_grads = torch.cat(grads)
        threshold = torch.quantile(all_grads, percentile / 100.0)

        for p in parameters:
            if p.grad is not None:
                p.grad.data.clamp_(-threshold, threshold)

        return threshold
```

#### 2.3.2 梯度噪声与正则化

```python
class GradientNoiseAndRegularization:
    """
    梯度噪声注入与正则化技术
    """

    def gradient_noise_injection(self, parameters, noise_level=0.01):
        """
        梯度噪声注入
        理论：帮助逃离尖锐最小值
        """
        for p in parameters:
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_level
                p.grad.data.add_(noise)

    def gradient_smoothing(self, parameters, momentum=0.9):
        """
        梯度平滑（指数移动平均）
        """
        if not hasattr(self, 'smooth_grads'):
            self.smooth_grads = {}

        for i, p in enumerate(parameters):
            if p.grad is not None:
                if i not in self.smooth_grads:
                    self.smooth_grads[i] = torch.zeros_like(p.grad)

                # 指数移动平均
                self.smooth_grads[i].mul_(momentum).add_(
                    p.grad.data, alpha=1 - momentum
                )

                # 替换原始梯度
                p.grad.data = self.smooth_grads[i]

    def sam_gradient(self, loss_fn, parameters, rho=0.05):
        """
        Sharpness-Aware Minimization (SAM)
        寻找平坦最小值
        """
        # 1. 计算原始梯度
        loss = loss_fn()
        loss.backward()

        # 保存原始梯度
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in parameters if p.grad is not None])
        )

        # 2. 在梯度方向扰动参数
        epsilon = rho / (grad_norm + 1e-12)
        for p in parameters:
            if p.grad is not None:
                e_w = p.grad * epsilon
                p.add_(e_w)  # 扰动

        # 3. 计算扰动后的梯度
        loss_fn().backward()

        # 4. 恢复原始参数
        for p in parameters:
            if p.grad is not None:
                e_w = p.grad * epsilon
                p.sub_(e_w)  # 恢复
```

---

## 第三部分：实战案例

### 3.1 自定义层与操作

```python
class CustomLayerExample:
    """
    完整的自定义层实现示例
    """

    @staticmethod
    def create_gaussian_rbf_layer():
        """
        高斯径向基函数（RBF）层
        """
        class GaussianRBF(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features

                # 中心点参数
                self.centers = torch.nn.Parameter(
                    torch.randn(out_features, in_features)
                )

                # 宽度参数
                self.log_sigma = torch.nn.Parameter(
                    torch.zeros(out_features)
                )

            def forward(self, x):
                """
                RBF: φ(x) = exp(-||x - c||² / (2σ²))
                """
                # x: (batch, in_features)
                # centers: (out_features, in_features)

                # 计算距离：(batch, out_features)
                x_expanded = x.unsqueeze(1)  # (batch, 1, in_features)
                centers_expanded = self.centers.unsqueeze(0)  # (1, out_features, in_features)

                distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=-1)

                # 应用RBF
                sigma = torch.exp(self.log_sigma)
                rbf_output = torch.exp(-distances / (2 * sigma ** 2).unsqueeze(0))

                return rbf_output

        return GaussianRBF

    @staticmethod
    def create_attention_layer():
        """
        多头注意力层（从零实现）
        """
        class MultiHeadAttention(torch.nn.Module):
            def __init__(self, d_model, num_heads):
                super().__init__()
                assert d_model % num_heads == 0

                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads

                # 权重矩阵
                self.W_q = torch.nn.Linear(d_model, d_model)
                self.W_k = torch.nn.Linear(d_model, d_model)
                self.W_v = torch.nn.Linear(d_model, d_model)
                self.W_o = torch.nn.Linear(d_model, d_model)

            def split_heads(self, x):
                """分割成多头"""
                batch_size, seq_len, d_model = x.shape
                return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            def forward(self, query, key, value, mask=None):
                """
                Attention(Q,K,V) = softmax(QK^T/√d_k)V
                """
                batch_size = query.shape[0]

                # 线性变换
                Q = self.split_heads(self.W_q(query))  # (batch, heads, seq, d_k)
                K = self.split_heads(self.W_k(key))
                V = self.split_heads(self.W_v(value))

                # 计算注意力分数
                scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
                    torch.tensor(self.d_k, dtype=torch.float32)
                )

                # 应用掩码
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))

                # Softmax
                attn_weights = torch.softmax(scores, dim=-1)

                # 加权求和
                output = torch.matmul(attn_weights, V)

                # 合并多头
                output = output.transpose(1, 2).contiguous().view(
                    batch_size, -1, self.d_model
                )

                # 输出投影
                return self.W_o(output)

        return MultiHeadAttention
```

### 3.2 性能优化实战

```python
class PerformanceOptimizationExamples:
    """
    性能优化实战案例
    """

    def optimize_batch_operations(self):
        """
        批量操作优化
        """
        # 低效：循环处理
        def process_slow(data_list):
            results = []
            for data in data_list:
                result = torch.sin(data) + torch.cos(data) ** 2
                results.append(result)
            return torch.stack(results)

        # 高效：批量处理
        def process_fast(data_tensor):
            return torch.sin(data_tensor) + torch.cos(data_tensor) ** 2

        # 基准测试
        data_list = [torch.randn(100, 100) for _ in range(10)]
        data_tensor = torch.stack(data_list)

        import time

        start = time.time()
        _ = process_slow(data_list)
        slow_time = time.time() - start

        start = time.time()
        _ = process_fast(data_tensor)
        fast_time = time.time() - start

        print(f"慢速: {slow_time:.4f}s, 快速: {fast_time:.4f}s")
        print(f"加速比: {slow_time / fast_time:.2f}x")

    def memory_efficient_attention(self, query, key, value):
        """
        内存高效的注意力计算
        FlashAttention风格
        """
        batch_size, seq_len, d_model = query.shape

        # 分块处理
        block_size = 128
        num_blocks = (seq_len + block_size - 1) // block_size

        output = torch.zeros_like(query)
        output_scale = torch.zeros(batch_size, seq_len, 1, device=query.device)

        for i in range(num_blocks):
            start_q = i * block_size
            end_q = min((i + 1) * block_size, seq_len)

            q_block = query[:, start_q:end_q, :]

            for j in range(num_blocks):
                start_k = j * block_size
                end_k = min((j + 1) * block_size, seq_len)

                k_block = key[:, start_k:end_k, :]
                v_block = value[:, start_k:end_k, :]

                # 计算分块注意力
                scores = torch.matmul(q_block, k_block.transpose(-2, -1))
                scores = scores / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

                attn_weights = torch.softmax(scores, dim=-1)
                block_output = torch.matmul(attn_weights, v_block)

                # 累积输出
                output[:, start_q:end_q, :] += block_output
                output_scale[:, start_q:end_q, :] += 1

        # 归一化
        output = output / output_scale
        return output
```

---

## 总结

本教程深入讲解了：

### 核心内容
1. **张量运算的数学基础**
   - 张量的数学定义与坐标变换
   - Einstein求和约定
   - 高性能张量操作技巧

2. **自动微分系统**
   - 计算图的内部实现
   - 梯度累积与选择性计算
   - 高阶导数计算

3. **优化技术**
   - 梯度裁剪策略
   - 梯度噪声与正则化
   - 内存高效计算

### 实践技能
- 实现自定义autograd函数
- 优化内存使用
- 性能分析与优化

### 下一步
继续学习**教程三：神经网络架构设计**

---

## 练习题

1. 实现一个数值稳定的LogSoftmax函数
2. 使用einsum实现批量矩阵乘法
3. 实现梯度检查点的简化版本
4. 优化一个给定的慢速PyTorch代码
5. 实现SAM优化器的简化版本
