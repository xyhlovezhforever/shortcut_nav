# PyTorch深度教程（四）：优化器与训练技巧

> **前置要求**：完成前三篇教程
> **核心目标**：掌握优化理论与工程最佳实践

---

## 第一部分：优化理论基础

### 1.1 梯度下降的数学原理

#### 1.1.1 一阶优化方法

```python
"""
梯度下降（Gradient Descent）

目标：最小化 f(θ)

更新规则：
θ_{t+1} = θ_t - α∇f(θ_t)

其中 α 是学习率
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class GradientDescentTheory:
    """梯度下降理论"""

    def vanilla_gradient_descent(self):
        """
        标准梯度下降（批量梯度下降）
        """
        def f(x):
            """目标函数：f(x) = x^2"""
            return x ** 2

        def grad_f(x):
            """梯度：f'(x) = 2x"""
            return 2 * x

        # 参数
        x = torch.tensor(10.0, requires_grad=True)
        learning_rate = 0.1
        num_iterations = 50

        # 梯度下降
        history = []
        for _ in range(num_iterations):
            loss = f(x)
            history.append((x.item(), loss.item()))

            # 计算梯度
            loss.backward()

            # 更新参数
            with torch.no_grad():
                x -= learning_rate * x.grad
                x.grad.zero_()

        print(f"最终x: {x.item():.6f}")
        print(f"最终loss: {f(x).item():.6f}")

    def stochastic_gradient_descent(self):
        """
        随机梯度下降（SGD）

        每次使用单个样本的梯度：
        θ_{t+1} = θ_t - α∇f(θ_t; x_i, y_i)

        优势：
        1. 计算快速
        2. 可以在线学习
        3. 噪声帮助逃离局部最小值

        劣势：
        1. 梯度估计有噪声
        2. 收敛不稳定
        """
        # 生成数据
        X = torch.randn(1000, 10)
        y = torch.randn(1000, 1)

        # 模型
        model = nn.Linear(10, 1)
        criterion = nn.MSELoss()

        # SGD优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 训练
        for epoch in range(100):
            for i in range(len(X)):
                # 单个样本
                x_i = X[i:i+1]
                y_i = y[i:i+1]

                # 前向传播
                pred = model(x_i)
                loss = criterion(pred, y_i)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def mini_batch_gradient_descent(self):
        """
        小批量梯度下降（Mini-batch GD）

        使用小批量样本的平均梯度：
        θ_{t+1} = θ_t - α(1/B)Σᵢ∇f(θ_t; x_i, y_i)

        折中：
        - 批量大小B：计算效率与梯度准确性的权衡
        - 典型值：32, 64, 128, 256
        """
        # 数据加载器
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 10),
            torch.randn(1000, 1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )

        model = nn.Linear(10, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(100):
            for batch_x, batch_y in dataloader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def convergence_analysis(self):
        """
        收敛性分析

        对于L-smooth、μ-strongly convex函数：
        条件数 κ = L/μ

        收敛率：
        - GD: O((1 - μ/L)^t)
        - 最优学习率: α = 1/L
        - 条件数越小，收敛越快
        """
        # 二次函数：f(x) = 0.5 * x^T Q x
        Q = torch.tensor([[2.0, 0.0], [0.0, 20.0]])  # 条件数 = 10

        # 特征值
        eigenvalues = torch.linalg.eigvalsh(Q)
        L = eigenvalues.max()  # 最大特征值
        mu = eigenvalues.min()  # 最小特征值
        kappa = L / mu  # 条件数

        print(f"L-smooth常数: {L}")
        print(f"强凸参数: {mu}")
        print(f"条件数: {kappa}")

        # 理论收敛率
        optimal_lr = 1 / L
        convergence_rate = 1 - mu / L

        print(f"最优学习率: {optimal_lr}")
        print(f"收敛率: {convergence_rate}")
```

### 1.2 动量方法

#### 1.2.1 动量（Momentum）

```python
class MomentumOptimization:
    """动量优化方法"""

    def momentum_theory(self):
        """
        动量方法

        更新规则：
        v_{t+1} = βv_t + ∇f(θ_t)
        θ_{t+1} = θ_t - αv_{t+1}

        物理类比：
        - v: 速度
        - ∇f: 力
        - β: 摩擦系数（典型值0.9）

        优势：
        1. 加速相关方向
        2. 减少振荡
        3. 帮助逃离局部最小值
        """

        class MomentumOptimizer:
            """从零实现Momentum"""
            def __init__(self, parameters, lr=0.01, momentum=0.9):
                self.parameters = list(parameters)
                self.lr = lr
                self.momentum = momentum
                self.velocities = [torch.zeros_like(p.data) for p in self.parameters]

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

    def nesterov_momentum(self):
        """
        Nesterov加速梯度（NAG）

        更新规则：
        v_{t+1} = βv_t + ∇f(θ_t - αβv_t)  # 前瞻梯度
        θ_{t+1} = θ_t - αv_{t+1}

        洞察：在预测的未来位置计算梯度

        优势：
        - 更好的收敛保证
        - 减少过冲
        """

        class NesterovOptimizer:
            """从零实现Nesterov Momentum"""
            def __init__(self, parameters, lr=0.01, momentum=0.9):
                self.parameters = list(parameters)
                self.lr = lr
                self.momentum = momentum
                self.velocities = [torch.zeros_like(p.data) for p in self.parameters]

            def step(self):
                with torch.no_grad():
                    for p, v in zip(self.parameters, self.velocities):
                        if p.grad is None:
                            continue

                        # Nesterov更新
                        v.mul_(self.momentum).add_(p.grad)
                        p.add_(v, alpha=-self.lr)

        # PyTorch内置
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            nesterov=True
        )

    def heavy_ball_vs_nesterov(self):
        """
        Heavy Ball vs Nesterov 对比
        """
        def rosenbrock(x):
            """Rosenbrock函数（病态优化问题）"""
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        # 初始点
        x_hb = torch.tensor([0.0, 0.0], requires_grad=True)
        x_nag = torch.tensor([0.0, 0.0], requires_grad=True)

        # 优化器
        opt_hb = torch.optim.SGD([x_hb], lr=0.001, momentum=0.9, nesterov=False)
        opt_nag = torch.optim.SGD([x_nag], lr=0.001, momentum=0.9, nesterov=True)

        # 比较收敛
        for _ in range(1000):
            # Heavy Ball
            loss_hb = rosenbrock(x_hb)
            opt_hb.zero_grad()
            loss_hb.backward()
            opt_hb.step()

            # Nesterov
            loss_nag = rosenbrock(x_nag)
            opt_nag.zero_grad()
            loss_nag.backward()
            opt_nag.step()

        print(f"Heavy Ball: {x_hb}, loss={rosenbrock(x_hb).item()}")
        print(f"Nesterov: {x_nag}, loss={rosenbrock(x_nag).item()}")
```

### 1.3 自适应学习率方法

#### 1.3.1 AdaGrad

```python
class AdaptiveLearningRateMethods:
    """自适应学习率优化器"""

    def adagrad_theory(self):
        """
        AdaGrad (Adaptive Gradient)

        更新规则：
        g_t = ∇f(θ_t)
        G_t = G_{t-1} + g_t^2  # 累积梯度平方
        θ_{t+1} = θ_t - α/√(G_t + ε) * g_t

        特点：
        1. 自动调整每个参数的学习率
        2. 稀疏特征得到更大更新
        3. 学习率单调递减

        问题：
        - 学习率过度衰减，可能过早停止
        """

        class AdaGrad:
            """从零实现AdaGrad"""
            def __init__(self, parameters, lr=0.01, eps=1e-8):
                self.parameters = list(parameters)
                self.lr = lr
                self.eps = eps
                self.sum_squared_grads = [torch.zeros_like(p.data) for p in self.parameters]

            def step(self):
                with torch.no_grad():
                    for p, g_sum in zip(self.parameters, self.sum_squared_grads):
                        if p.grad is None:
                            continue

                        # 累积梯度平方
                        g_sum.add_(p.grad ** 2)

                        # 自适应学习率
                        adapted_lr = self.lr / (torch.sqrt(g_sum) + self.eps)

                        # 更新参数
                        p.add_(p.grad * adapted_lr, alpha=-1)

    def rmsprop_theory(self):
        """
        RMSProp (Root Mean Square Propagation)

        更新规则：
        E[g^2]_t = γE[g^2]_{t-1} + (1-γ)g_t^2  # 指数移动平均
        θ_{t+1} = θ_t - α/√(E[g^2]_t + ε) * g_t

        改进：
        - 使用EMA代替累积和
        - 避免学习率过度衰减

        超参数：
        - γ: 衰减率（典型值0.9, 0.99）
        """

        class RMSProp:
            """从零实现RMSProp"""
            def __init__(self, parameters, lr=0.001, alpha=0.99, eps=1e-8):
                self.parameters = list(parameters)
                self.lr = lr
                self.alpha = alpha
                self.eps = eps
                self.square_avg = [torch.zeros_like(p.data) for p in self.parameters]

            def step(self):
                with torch.no_grad():
                    for p, avg in zip(self.parameters, self.square_avg):
                        if p.grad is None:
                            continue

                        # 指数移动平均
                        avg.mul_(self.alpha).addcmul_(
                            p.grad, p.grad, value=1 - self.alpha
                        )

                        # 自适应学习率
                        adapted_lr = self.lr / (torch.sqrt(avg) + self.eps)

                        # 更新参数
                        p.add_(p.grad * adapted_lr, alpha=-1)

    def adam_theory(self):
        """
        Adam (Adaptive Moment Estimation)

        结合Momentum和RMSProp

        更新规则：
        m_t = β₁m_{t-1} + (1-β₁)g_t          # 一阶矩估计
        v_t = β₂v_{t-1} + (1-β₂)g_t^2        # 二阶矩估计

        # 偏差修正
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)

        θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)

        超参数：
        - α: 学习率（典型值0.001）
        - β₁: 一阶矩衰减（典型值0.9）
        - β₂: 二阶矩衰减（典型值0.999）
        - ε: 数值稳定性（典型值1e-8）
        """

        class Adam:
            """从零实现Adam"""
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

    def adamw_theory(self):
        """
        AdamW (Adam with Weight Decay)

        正确的权重衰减实现

        更新规则：
        θ_{t+1} = θ_t - α(m̂_t/(√v̂_t + ε) + λθ_t)

        其中λ是权重衰减系数

        与L2正则化的区别：
        - L2正则化：在损失中添加 λ||θ||²
        - 权重衰减：直接在更新中减去 λθ

        对于Adam，两者不等价！AdamW更正确
        """

        # PyTorch实现
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01  # 权重衰减
        )

    def optimizer_comparison(self):
        """
        优化器对比实验
        """
        # 测试函数：Beale函数（多模态）
        def beale(x, y):
            return ((1.5 - x + x*y)**2 +
                    (2.25 - x + x*y**2)**2 +
                    (2.625 - x + x*y**3)**2)

        optimizers = {
            'SGD': torch.optim.SGD,
            'Momentum': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9),
            'AdaGrad': torch.optim.Adagrad,
            'RMSProp': torch.optim.RMSprop,
            'Adam': torch.optim.Adam,
        }

        results = {}
        for name, opt_class in optimizers.items():
            x = torch.tensor([0.0, 0.0], requires_grad=True)
            optimizer = opt_class([x], lr=0.001) if callable(opt_class) else opt_class([x])

            trajectory = []
            for _ in range(1000):
                loss = beale(x[0], x[1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trajectory.append((x[0].item(), x[1].item()))

            results[name] = trajectory

        # 可视化轨迹...
```

---

## 第二部分：学习率调度

### 2.1 学习率调度策略

```python
class LearningRateScheduling:
    """学习率调度"""

    def step_decay(self):
        """
        步阶衰减（Step Decay）

        每隔固定epoch降低学习率
        α_t = α_0 * γ^⌊t/step_size⌋
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,  # 每30个epoch
            gamma=0.1      # 乘以0.1
        )

        for epoch in range(100):
            # 训练...
            scheduler.step()

    def multi_step_decay(self):
        """
        多步衰减（Multi-Step Decay）

        在指定的epoch降低学习率
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],  # 在这些epoch降低
            gamma=0.1
        )

    def exponential_decay(self):
        """
        指数衰减（Exponential Decay）

        α_t = α_0 * γ^t
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # 每epoch乘以0.95
        )

    def cosine_annealing(self):
        """
        余弦退火（Cosine Annealing）

        α_t = α_min + (α_max - α_min) * (1 + cos(πt/T)) / 2

        其中T是总epoch数

        特点：
        - 平滑衰减
        - 初期快速下降
        - 后期缓慢趋近最小值
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,    # 周期
            eta_min=1e-5  # 最小学习率
        )

    def cosine_annealing_warm_restarts(self):
        """
        带热重启的余弦退火（SGDR）

        周期性重启学习率
        帮助逃离局部最小值

        α_t = α_min + (α_max - α_min) * (1 + cos(πt_cur/T_cur)) / 2

        其中t_cur在每个重启时重置
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,        # 初始周期
            T_mult=2,      # 周期倍增因子
            eta_min=1e-5
        )

        # 每个epoch调用
        for epoch in range(100):
            # 训练...
            scheduler.step()

    def reduce_on_plateau(self):
        """
        自适应衰减（ReduceLROnPlateau）

        当指标停止改善时降低学习率
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',        # 'min'表示最小化指标
            factor=0.1,        # 衰减因子
            patience=10,       # 容忍的epoch数
            threshold=1e-4,    # 改善的最小阈值
            min_lr=1e-6        # 最小学习率
        )

        for epoch in range(100):
            # 训练...
            val_loss = validate()
            scheduler.step(val_loss)  # 传入验证损失

    def warmup_schedule(self):
        """
        预热（Warmup）

        训练初期线性增加学习率
        防止初期梯度过大导致不稳定

        常用于Transformer等大模型
        """
        class WarmupScheduler:
            def __init__(self, optimizer, warmup_steps, d_model):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.d_model = d_model
                self.step_num = 0

            def step(self):
                self.step_num += 1

                # Transformer论文中的公式
                lr = self.d_model ** (-0.5) * min(
                    self.step_num ** (-0.5),
                    self.step_num * self.warmup_steps ** (-1.5)
                )

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        # 使用
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = WarmupScheduler(optimizer, warmup_steps=4000, d_model=512)

    def one_cycle_policy(self):
        """
        单周期策略（1cycle Policy）

        Leslie Smith提出，包括：
        1. Warmup阶段
        2. Annealing阶段

        学习率和动量反向变化
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1,           # 最大学习率
            steps_per_epoch=len(train_loader),
            epochs=100,
            pct_start=0.3,        # warmup占比
            anneal_strategy='cos' # 'cos' or 'linear'
        )

        for epoch in range(100):
            for batch in train_loader:
                # 训练...
                scheduler.step()  # 每个batch调用
```

### 2.2 学习率查找

```python
class LearningRateFinder:
    """学习率查找器"""

    def lr_range_test(self, model, train_loader, min_lr=1e-7, max_lr=10, num_iter=100):
        """
        学习率范围测试（LR Range Test）

        Leslie Smith的方法：
        1. 从极小学习率开始
        2. 指数增长到极大学习率
        3. 记录每个学习率对应的损失
        4. 选择损失下降最快的学习率
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
        criterion = nn.CrossEntropyLoss()

        # 学习率倍增因子
        lr_mult = (max_lr / min_lr) ** (1 / num_iter)

        # 记录
        lrs = []
        losses = []

        # 保存初始状态
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        model.train()
        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0

        iterator = iter(train_loader)
        for iteration in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            batch_num += 1

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 计算平滑损失
            avg_loss = 0.98 * avg_loss + 0.02 * loss.item() if batch_num > 1 else loss.item()

            # 记录
            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(avg_loss)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 增加学习率
            optimizer.param_groups[0]['lr'] *= lr_mult

            # 发散检测
            if avg_loss > 4 * best_loss:
                break

            if avg_loss < best_loss:
                best_loss = avg_loss

        # 恢复模型状态
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        # 绘制曲线
        import matplotlib.pyplot as plt
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.show()

        # 推荐学习率：损失下降最快处
        # 通常选择最陡下降点的1/10
        gradients = np.gradient(losses)
        best_lr = lrs[np.argmin(gradients)]
        suggested_lr = best_lr / 10

        print(f"建议学习率: {suggested_lr}")

        return lrs, losses
```

---

## 第三部分：训练技巧

### 3.1 批归一化与训练稳定性

```python
class TrainingStabilityTechniques:
    """训练稳定性技术"""

    def gradient_clipping_strategies(self):
        """
        梯度裁剪策略
        """

        # 1. 按值裁剪
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1.0, 1.0)

        # 2. 按范数裁剪（推荐）
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # 3. 按全局范数裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            norm_type=2.0  # L2范数
        )

    def mixed_precision_training(self):
        """
        混合精度训练

        使用FP16加速训练，同时保持FP32精度

        优势：
        1. 减少内存使用
        2. 加速计算（Tensor Cores）
        3. 允许更大batch size
        """
        from torch.cuda.amp import autocast, GradScaler

        model = nn.Linear(100, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()

        for epoch in range(100):
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()

                # 自动混合精度上下文
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # 缩放损失并反向传播
                scaler.scale(loss).backward()

                # 梯度裁剪（可选）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()

    def gradient_accumulation(self):
        """
        梯度累积

        模拟大batch size训练
        """
        accumulation_steps = 4
        effective_batch_size = batch_size * accumulation_steps

        optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 归一化损失
            loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            # 累积足够步数后更新
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    def label_smoothing(self):
        """
        标签平滑（Label Smoothing）

        软化one-hot标签，防止过拟合

        y_smooth = (1 - ε) * y_true + ε / K

        其中：
        - ε: 平滑因子（典型值0.1）
        - K: 类别数
        """
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, pred, target):
                """
                pred: (batch, num_classes) logits
                target: (batch,) class indices
                """
                num_classes = pred.shape[-1]
                log_probs = torch.log_softmax(pred, dim=-1)

                # 平滑后的标签
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_probs)
                    true_dist.fill_(self.smoothing / (num_classes - 1))
                    true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

                return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

    def stochastic_depth(self):
        """
        随机深度（Stochastic Depth）

        训练时随机丢弃残差块
        """
        class StochasticDepthBlock(nn.Module):
            def __init__(self, module, survival_prob=0.9):
                super().__init__()
                self.module = module
                self.survival_prob = survival_prob

            def forward(self, x):
                if not self.training:
                    return x + self.module(x)

                # 训练时随机丢弃
                if torch.rand(1) < self.survival_prob:
                    return x + self.module(x)
                else:
                    return x

    def exponential_moving_average(self):
        """
        指数移动平均（EMA）

        维护参数的移动平均
        推理时使用平均参数
        """
        class EMA:
            def __init__(self, model, decay=0.999):
                self.model = model
                self.decay = decay
                self.shadow = {}
                self.backup = {}

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()

            def update(self):
                """更新影子参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = self.decay * self.shadow[name] + \
                                           (1 - self.decay) * param.data

            def apply_shadow(self):
                """应用影子参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.backup[name] = param.data
                        param.data = self.shadow[name]

            def restore(self):
                """恢复原始参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = self.backup[name]
                self.backup = {}

        # 使用
        ema = EMA(model, decay=0.999)

        for epoch in range(100):
            for batch in train_loader:
                # 训练
                loss = train_step(batch)

                # 更新EMA
                ema.update()

            # 验证时使用EMA参数
            ema.apply_shadow()
            val_loss = validate()
            ema.restore()
```

### 3.2 正则化技术

```python
class RegularizationTechniques:
    """正则化技术"""

    def weight_decay_theory(self):
        """
        权重衰减（Weight Decay）

        L2正则化：
        L_reg = L + λ/2 * ||θ||²

        梯度：
        ∇L_reg = ∇L + λθ

        更新：
        θ ← θ - α(∇L + λθ)
          = (1 - αλ)θ - α∇L
        """
        # PyTorch实现
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            weight_decay=1e-4  # λ
        )

    def dropout_theory(self):
        """
        Dropout

        训练时：
        - 以概率p丢弃神经元
        - 剩余神经元输出放大1/(1-p)

        推理时：
        - 使用所有神经元
        - 输出不变

        效果：
        - 集成学习（多个子网络）
        - 减少共适应（co-adaptation）
        """
        class DropoutLayer(nn.Module):
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p

            def forward(self, x):
                if not self.training:
                    return x

                # 生成掩码
                mask = (torch.rand_like(x) > self.p).float()

                # 缩放
                return x * mask / (1 - self.p)

        # 变体：DropConnect
        class DropConnect(nn.Module):
            def __init__(self, linear_layer, p=0.5):
                super().__init__()
                self.linear = linear_layer
                self.p = p

            def forward(self, x):
                if not self.training:
                    return self.linear(x)

                # 丢弃权重而非激活
                weight = self.linear.weight
                mask = (torch.rand_like(weight) > self.p).float()
                masked_weight = weight * mask / (1 - self.p)

                return torch.nn.functional.linear(x, masked_weight, self.linear.bias)

    def data_augmentation(self):
        """
        数据增强

        图像：
        - 随机裁剪、翻转
        - 颜色抖动
        - Cutout、Mixup、CutMix
        """
        from torchvision import transforms

        # 标准增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Cutout
        class Cutout:
            def __init__(self, n_holes=1, length=16):
                self.n_holes = n_holes
                self.length = length

            def __call__(self, img):
                h, w = img.shape[1:]
                mask = torch.ones_like(img)

                for _ in range(self.n_holes):
                    y = torch.randint(h, (1,))
                    x = torch.randint(w, (1,))

                    y1 = torch.clamp(y - self.length // 2, 0, h)
                    y2 = torch.clamp(y + self.length // 2, 0, h)
                    x1 = torch.clamp(x - self.length // 2, 0, w)
                    x2 = torch.clamp(x - self.length // 2, 0, w)

                    mask[:, y1:y2, x1:x2] = 0

                return img * mask

        # Mixup
        def mixup_data(x, y, alpha=1.0):
            """
            混合两个样本

            x̃ = λx_i + (1-λ)x_j
            ỹ = λy_i + (1-λ)y_j

            其中 λ ~ Beta(α, α)
            """
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = x.shape[0]
            index = torch.randperm(batch_size)

            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]

            return mixed_x, y_a, y_b, lam

        def mixup_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### 3.3 模型评估与选择

```python
class ModelEvaluationAndSelection:
    """模型评估与选择"""

    def cross_validation(self):
        """
        交叉验证（Cross-Validation）

        K折交叉验证：
        1. 将数据分成K份
        2. 每次用K-1份训练，1份验证
        3. 重复K次
        4. 平均结果
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32)
            val_loader = DataLoader(val_subset, batch_size=32)

            # 创建新模型
            model = create_model()

            # 训练
            for epoch in range(100):
                train(model, train_loader)

            # 验证
            score = evaluate(model, val_loader)
            scores.append(score)

        print(f"平均得分: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    def early_stopping(self):
        """
        早停（Early Stopping）

        监控验证损失，防止过拟合
        """
        class EarlyStopping:
            def __init__(self, patience=7, min_delta=0, mode='min'):
                self.patience = patience
                self.min_delta = min_delta
                self.mode = mode
                self.counter = 0
                self.best_score = None
                self.early_stop = False

            def __call__(self, val_loss):
                score = -val_loss if self.mode == 'min' else val_loss

                if self.best_score is None:
                    self.best_score = score
                elif score < self.best_score + self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.counter = 0

        # 使用
        early_stopping = EarlyStopping(patience=10)

        for epoch in range(1000):
            train_loss = train_epoch()
            val_loss = validate_epoch()

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    def model_checkpoint(self):
        """
        模型检查点（Model Checkpoint）

        保存最佳模型
        """
        class ModelCheckpoint:
            def __init__(self, filepath, monitor='val_loss', mode='min'):
                self.filepath = filepath
                self.monitor = monitor
                self.mode = mode
                self.best_score = None

            def __call__(self, model, val_loss):
                score = -val_loss if self.mode == 'min' else val_loss

                if self.best_score is None or score > self.best_score:
                    self.best_score = score

                    # 保存模型
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'score': score,
                    }, self.filepath)

                    print(f"Saved best model with {self.monitor}={val_loss:.4f}")

        # 使用
        checkpoint = ModelCheckpoint('best_model.pth', monitor='val_loss')

        for epoch in range(100):
            train_epoch()
            val_loss = validate_epoch()
            checkpoint(model, val_loss)
```

---

## 总结

本教程涵盖了PyTorch训练的核心内容：

### 优化理论
- 梯度下降及其变体
- 动量方法
- 自适应学习率优化器

### 学习率策略
- 各种调度方法
- 学习率查找
- Warmup和退火

### 训练技巧
- 混合精度训练
- 梯度裁剪
- 正则化技术

### 模型评估
- 交叉验证
- 早停
- 模型检查点

### 下一步
继续学习**教程五：分布式训练与性能优化**
