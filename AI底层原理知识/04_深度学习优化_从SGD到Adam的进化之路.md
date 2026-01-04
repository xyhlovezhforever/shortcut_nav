# 深度学习优化:从 SGD 到 Adam 的进化之路

## 1. 引言

优化算法是深度学习的"心脏",决定了模型能否有效训练。从最简单的 SGD 到现代的 Adam,每一次进化都在解决前一代算法的痛点。本教程深入剖析主流优化器的数学原理、设计动机和实战技巧。

---

## 2. 优化问题的本质

### 2.1 目标函数景观

深度学习的优化目标:
```
min L(θ) = (1/n) Σ loss(f(x_i; θ), y_i)
θ                i
```

**挑战:**
1. **非凸性:** 大量局部极小值和鞍点
2. **高维性:** 参数可达数十亿(GPT-3: 175B)
3. **病态曲率:** 不同方向曲率差异大(Hessian 特征值范围广)
4. **噪声梯度:** Mini-batch 梯度是真实梯度的无偏但有噪声估计

### 2.2 优化 vs 泛化

**训练误差 vs 测试误差:**
```
L_train(θ) ≠ L_test(θ)
```

**Sharp vs Flat Minima:**
- Sharp: 损失函数陡峭,泛化差
- Flat: 损失函数平坦,泛化好

**大 Batch Size 的问题:** 倾向于收敛到 sharp minima

---

## 3. 梯度下降的变体

### 3.1 Batch Gradient Descent (BGD)

**更新规则:**
```
θ_(t+1) = θ_t - η · ∇L(θ_t)
```

其中:
```
∇L(θ_t) = (1/n) Σ ∇loss(f(x_i; θ_t), y_i)
                 i=1
```

**代码:**
```python
def batch_gradient_descent(X, y, theta, lr, epochs):
    for epoch in range(epochs):
        grad = compute_gradient(X, y, theta)  # 整个数据集
        theta -= lr * grad
    return theta
```

**特点:**
- ✓ 梯度准确
- ✓ 收敛稳定
- ✗ 计算慢(每步需遍历整个数据集)
- ✗ 内存需求大
- ✗ 无法在线学习

---

### 3.2 Stochastic Gradient Descent (SGD)

**更新规则:**
```
θ_(t+1) = θ_t - η · ∇loss(f(x_i; θ_t), y_i)
```

每次只用**一个样本**。

**代码:**
```python
def stochastic_gradient_descent(X, y, theta, lr, epochs):
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            grad = compute_gradient(X[i:i+1], y[i:i+1], theta)
            theta -= lr * grad
    return theta
```

**特点:**
- ✓ 更新快
- ✓ 可能逃离局部极小值(噪声)
- ✓ 在线学习
- ✗ 梯度噪声大
- ✗ 收敛不稳定

**数学性质:**
```
E[∇loss(x_i)] = ∇L(θ)       # 无偏估计
Var[∇loss(x_i)] = σ²        # 方差不为 0
```

---

### 3.3 Mini-Batch Gradient Descent

**更新规则:**
```
θ_(t+1) = θ_t - η · (1/B) Σ ∇loss(f(x_i; θ_t), y_i)
                          i∈B
```

B: batch size (常用 32, 64, 128, 256)

**代码:**
```python
def mini_batch_gradient_descent(X, y, theta, lr, batch_size, epochs):
    n = len(X)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            grad = compute_gradient(X_batch, y_batch, theta)
            theta -= lr * grad
    return theta
```

**梯度方差:**
```
Var[∇L_B] = σ² / B
```

**特点:**
- ✓ 平衡速度与稳定性
- ✓ GPU 并行加速
- ✓ 方差随 batch size 减小
- ⚖ Batch size 选择是艺术

---

### 3.4 Batch Size 的影响

| Batch Size | 优势 | 劣势 |
|------------|------|------|
| 小 (32-64) | 泛化好,收敛到 flat minima | 训练慢,不稳定 |
| 中 (128-256) | 平衡 | - |
| 大 (1024+) | 训练快,GPU 利用率高 | 泛化差,需调整学习率 |

**线性缩放规则:**
增大 batch size B 倍,学习率也增大 B 倍。
```
lr_new = lr_old * (B_new / B_old)
```

**推导:** 保持参数更新步长不变
```
Δθ = -η · ∇L_B
```

---

## 4. Momentum 系列

### 4.1 经典 Momentum

**动机:** SGD 在平坦方向慢,陡峭方向震荡。

**物理类比:** 小球滚下山坡,累积动量。

**更新规则:**
```
v_t = β·v_(t-1) + ∇L(θ_t)
θ_(t+1) = θ_t - η·v_t
```

**参数:**
- β ∈ [0, 1): 动量系数(常用 0.9)
- v_0 = 0

**展开式:**
```
v_t = ∇L_t + β·∇L_(t-1) + β²·∇L_(t-2) + ...
    = Σ β^k · ∇L_(t-k)
      k=0
```

**指数加权移动平均(EMA):**
```
v_t = (1-β) Σ β^k · ∇L_(t-k)
```

**代码:**
```python
class MomentumOptimizer:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i]
            params[i] -= self.lr * self.velocity[i]

        return params
```

**效果:**
- ✓ 加速收敛(平坦方向累积)
- ✓ 减少震荡(陡峭方向抵消)
- ✓ 逃离局部极小值

**衰减系数的含义:**
β=0.9 意味着大约保留过去 1/(1-β)=10 步的梯度信息。

---

### 4.2 Nesterov Accelerated Gradient (NAG)

**动机:** Momentum 盲目跟随梯度,可能错过转折。

**核心思想:** "先跳跃,再修正"
```
1. 预测位置: θ̃ = θ_t - β·v_(t-1)
2. 在预测位置计算梯度: ∇L(θ̃)
3. 更新动量: v_t = β·v_(t-1) + ∇L(θ̃)
4. 更新参数: θ_(t+1) = θ_t - η·v_t
```

**对比:**
```
Momentum: v_t = β·v_(t-1) + ∇L(θ_t)
NAG:      v_t = β·v_(t-1) + ∇L(θ_t - β·v_(t-1))
```

**几何解释:** NAG 在动量方向上"向前看",提前感知梯度变化。

**代码:**
```python
class NesterovOptimizer:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self, params, compute_grad_fn):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        # 预测位置
        params_lookahead = [
            p - self.momentum * v for p, v in zip(params, self.velocity)
        ]

        # 在预测位置计算梯度
        grads = compute_grad_fn(params_lookahead)

        # 更新
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i]
            params[i] -= self.lr * self.velocity[i]

        return params
```

**收敛速率:**
- SGD: O(1/√T)
- Momentum: O(1/T) (强凸)
- NAG: O(1/T²) (强凸)

---

## 5. 自适应学习率方法

### 5.1 AdaGrad

**动机:** 不同参数需要不同学习率
- 稀疏特征(词频低): 需要大学习率
- 频繁特征(常见词): 需要小学习率

**更新规则:**
```
G_t = G_(t-1) + (∇L_t)²           # 累积梯度平方
θ_(t+1) = θ_t - η/√(G_t + ε) · ∇L_t
```

**逐元素:**
```
G_t,i = Σ (∂L/∂θ_i)²
        k=1

θ_(t+1,i) = θ_t,i - η/√(G_t,i + ε) · ∂L/∂θ_i
```

**代码:**
```python
class AdaGrad:
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G = None

    def update(self, params, grads):
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.G[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.G[i]) + self.epsilon)

        return params
```

**特点:**
- ✓ 自动调整学习率
- ✓ 适合稀疏数据
- ✗ G_t 单调增加,学习率不断减小
- ✗ 后期可能过早停止学习

**应用:** 词嵌入训练(Word2Vec, GloVe)

---

### 5.2 RMSProp

**动机:** 解决 AdaGrad 学习率衰减过快的问题

**核心改进:** 用指数加权移动平均代替累积和
```
E[g²]_t = β·E[g²]_(t-1) + (1-β)·(∇L_t)²
θ_(t+1) = θ_t - η/√(E[g²]_t + ε) · ∇L_t
```

**代码:**
```python
class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.E = None

    def update(self, params, grads):
        if self.E is None:
            self.E = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.E[i] = self.beta * self.E[i] + (1 - self.beta) * grads[i]**2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.E[i]) + self.epsilon)

        return params
```

**参数:**
- β = 0.9 (常用)
- η = 0.001

**特点:**
- ✓ 学习率不会单调递减
- ✓ 适合非平稳目标
- ✓ RNN 训练中表现好

**Hinton 在 Coursera 课程中提出,但未发表论文。**

---

### 5.3 Adam (Adaptive Moment Estimation)

**现代深度学习的默认选择!**

**核心思想:** 结合 Momentum 和 RMSProp
```
m_t = β1·m_(t-1) + (1-β1)·∇L_t          # 一阶矩(均值)
v_t = β2·v_(t-1) + (1-β2)·(∇L_t)²       # 二阶矩(未中心化方差)
```

**偏差修正:**
```
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
```

**参数更新:**
```
θ_(t+1) = θ_t - η · m̂_t / (√v̂_t + ε)
```

**为什么需要偏差修正?**

初始化 m_0 = 0, v_0 = 0 导致早期偏向 0:
```
m_1 = (1-β1)·∇L_1  ≪ E[∇L_1]  (若 β1 接近 1)
```

修正后:
```
m̂_1 = m_1 / (1-β1) = ∇L_1
```

**完整代码:**
```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            # 更新一阶矩和二阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # 更新参数
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
```

**标准超参数:**
- lr = 0.001
- β1 = 0.9
- β2 = 0.999
- ε = 1e-8

**优势:**
- ✓ 鲁棒性强,几乎总是有效
- ✓ 超参数不敏感
- ✓ 适合稀疏梯度
- ✓ 适合非平稳目标

**应用:** Transformer, GPT, BERT 等

---

### 5.4 AdamW (Adam with Weight Decay)

**动机:** 原始 Adam 中 L2 正则化与自适应学习率交互不良。

**L2 正则化:**
```
L'(θ) = L(θ) + λ/2 · ||θ||²
∇L'(θ) = ∇L(θ) + λ·θ
```

**问题:** Adam 的自适应学习率会缩放正则化项,导致效果减弱。

**解决方案 - Weight Decay:** 直接在参数更新时减小权重
```
θ_(t+1) = θ_t - η · m̂_t / (√v̂_t + ε) - η·λ·θ_t
```

**代码:**
```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Adam 更新 + 权重衰减
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]

        return params
```

**应用:** Transformer 训练的标配

---

## 6. 学习率调度策略

### 6.1 为什么需要调度?

**训练初期:** 需要大学习率快速下降
**训练后期:** 需要小学习率精细调整

### 6.2 常见策略

**(1) Step Decay**
```python
def step_decay(epoch, lr_init, drop_rate=0.5, epochs_drop=10):
    return lr_init * (drop_rate ** (epoch // epochs_drop))
```

**(2) Exponential Decay**
```python
def exp_decay(epoch, lr_init, decay_rate=0.95):
    return lr_init * (decay_rate ** epoch)
```

**(3) Cosine Annealing**
```python
def cosine_annealing(epoch, T_max, lr_min=0, lr_max=0.1):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max))
```

**优势:** 平滑过渡,避免突变

**(4) Warm-up + Cosine**
```python
def warmup_cosine(epoch, warmup_epochs, T_max, lr_max=0.1):
    if epoch < warmup_epochs:
        return lr_max * (epoch / warmup_epochs)
    else:
        return cosine_annealing(epoch - warmup_epochs, T_max - warmup_epochs, 0, lr_max)
```

**应用:** Transformer 训练(BERT, GPT)

---

### 6.3 Warm-up 的必要性

**问题:** Adam 等自适应方法在训练初期不稳定(偏差修正后仍有高方差)

**解决:** 前几个 epoch 线性增加学习率
```
lr_t = lr_max · min(1, t / T_warmup)
```

**典型配置:**
- T_warmup = 10% × 总步数

---

## 7. 优化器对比实验

### 7.1 实验设置

**任务:** 训练 MNIST 分类器

**网络:** 2 层 MLP (784 → 256 → 10)

**代码:**
```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为 one-hot
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_oh = to_one_hot(y_train)
y_test_oh = to_one_hot(y_test)

# 简单 MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.probs - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return [dW1, db1, dW2, db2]

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params

# 训练函数
def train(optimizer_class, optimizer_kwargs, epochs=20, batch_size=128):
    model = MLP(784, 256, 10)
    optimizer = optimizer_class(**optimizer_kwargs)

    train_losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train_oh[batch_idx]

            # 前向+反向
            probs = model.forward(X_batch)
            grads = model.backward(X_batch, y_batch)

            # 更新
            params = model.get_params()
            params = optimizer.update(params, grads)
            model.set_params(params)

        # 评估
        train_probs = model.forward(X_train[:5000])
        train_loss = -np.mean(np.sum(y_train_oh[:5000] * np.log(train_probs + 1e-8), axis=1))
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")

    return train_losses

# 运行对比
optimizers = {
    'SGD': (MomentumOptimizer, {'lr': 0.01, 'momentum': 0}),
    'Momentum': (MomentumOptimizer, {'lr': 0.01, 'momentum': 0.9}),
    'RMSProp': (RMSProp, {'lr': 0.001}),
    'Adam': (Adam, {'lr': 0.001}),
}

for name, (opt_class, opt_kwargs) in optimizers.items():
    print(f"\n训练 {name}...")
    losses = train(opt_class, opt_kwargs)
```

**结果分析:**
- SGD: 收敛慢,震荡
- Momentum: 比 SGD 快,仍震荡
- RMSProp: 稳定,较快
- Adam: 最快最稳定

---

## 8. 实战技巧

### 8.1 学习率选择

**方法 1: Grid Search**
```python
lrs = [1e-1, 1e-2, 1e-3, 1e-4]
for lr in lrs:
    train_with_lr(lr)
```

**方法 2: Learning Rate Finder**
```python
def lr_finder(model, X, y, lr_min=1e-7, lr_max=10, num_steps=100):
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), num_steps)
    losses = []

    for lr in lrs:
        optimizer = Adam(lr=lr)
        loss = train_one_step(model, X, y, optimizer)
        losses.append(loss)

    # 绘图
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

    # 选择损失下降最快的 lr
    gradients = np.gradient(losses)
    optimal_lr = lrs[np.argmin(gradients)]
    return optimal_lr
```

---

### 8.2 梯度裁剪(Gradient Clipping)

**问题:** 梯度爆炸(RNN 中常见)

**解决:**
```python
def clip_gradients(grads, max_norm=5.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = [g * clip_coef for g in grads]
    return grads
```

---

### 8.3 权重初始化

**Xavier 初始化(Sigmoid/Tanh):**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
```

**He 初始化(ReLU):**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

**原理:** 保持前向传播和反向传播的方差稳定。

---

## 9. 前沿优化器

### 9.1 Lookahead

**思想:** 维护两组权重,慢权重跟随快权重
```
θ_fast: 快速更新(如 Adam)
θ_slow: 每 k 步插值更新

θ_slow ← θ_slow + α(θ_fast - θ_slow)
```

### 9.2 LAMB (Layer-wise Adaptive Moments)

**动机:** 适应大 batch 训练(BERT 用 64K batch size)

**核心:** 逐层自适应信任域
```
r_t = ||θ_t|| / ||m̂_t / (√v̂_t + ε)||
θ_(t+1) = θ_t - η · r_t · m̂_t / (√v̂_t + ε)
```

### 9.3 AdaBelief

**改进 Adam:** 适应梯度的"置信度"
```
s_t = β2·s_(t-1) + (1-β2)·(∇L_t - m_t)²
θ_(t+1) = θ_t - η · m̂_t / (√ŝ_t + ε)
```

---

## 10. 总结

### 优化器选择指南

| 场景 | 推荐优化器 | 学习率 |
|------|-----------|--------|
| 默认选择 | Adam / AdamW | 1e-3 |
| CV (CNN) | SGD + Momentum | 0.1 (+ decay) |
| NLP (Transformer) | AdamW + Warmup | 1e-4 |
| 强化学习 | Adam | 3e-4 |
| GAN | Adam | 2e-4 (G), 2e-4 (D) |

### 关键洞察
1. **没有银弹:** 不同任务最优优化器不同
2. **Adam 的统治:** 90% 情况下 Adam 足够好
3. **学习率最重要:** 比选择优化器更关键
4. **Warmup + Decay:** 现代训练的标配

### 调试技巧
- 损失 NaN: 学习率过大,梯度爆炸
- 损失不动: 学习率过小,初始化问题
- 训练集过拟合: 加正则化,减小模型

**记住:** 优化是手段,泛化是目的!
