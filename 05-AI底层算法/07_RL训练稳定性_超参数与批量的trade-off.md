# RL训练稳定性：超参数与批量的权衡艺术

## 一、强化学习训练为什么不稳定？

### 1.1 三大不稳定来源

**1. 非平稳数据分布（Non-stationary Distribution）**
```
传统监督学习：数据分布固定
RL训练：策略改变 → 采样分布改变 → 训练目标移动
```

**案例**：
- 第1000步：模型倾向生成简短答案（平均长度2000）
- 第1001步：更新后模型开始生成长答案（平均长度5000）
- 第1002步：基于长答案的梯度又把模型拉回短答案
- 结果：长度在2000-5000之间震荡

**2. 高方差梯度（High Variance Gradients）**
```python
# 策略梯度公式
∇J = E[∇log π(a|s) · R(s,a)]

问题：不同轨迹的奖励差异巨大
- 轨迹1：R=1.0（答对）
- 轨迹2：R=0.0（答错）
→ 梯度方向完全相反！
```

**3. 奖励稀疏性（Sparse Rewards）**
```
数学题场景：
- 推理步骤1-50：没有反馈
- 最后一步：答案正确→+1，错误→0
→ 模型不知道哪个中间步骤有用
```

### 1.2 不稳定的表现形式

| 症状 | 表现 | 根本原因 |
|------|------|----------|
| **策略崩溃** | 奖励突然跌到0 | 更新步长过大 |
| **震荡不收敛** | 指标反复横跳 | 梯度方差大 |
| **过早收敛** | 熵过快降到0 | 探索不足 |
| **奖励欺骗** | 高奖励但质量差 | 奖励函数漏洞 |

---

## 二、JustRL的核心发现：批量放大是稳定性的关键

### 2.1 理论分析：为什么大批量稳定？

**梯度估计的方差公式**：
```
Var[∇J] = Var[∇log π(a|s) · R] / N

其中N是批量大小
→ 批量翻倍，方差减半
```

**实际影响**：

| 批量大小 | 梯度方差 | 训练曲线 | 需要调参程度 |
|---------|---------|---------|--------------|
| 64 | 高 | 剧烈震荡 | 需精细调整LR/KL |
| 256 | 中 | 小幅波动 | 需适度调整 |
| 512+ | 低 | 平滑上升 | 固定超参即可 |

### 2.2 实验验证：批量从64→512的变化

**实验设置**：
- 模型：DeepSeek-1.5B
- 任务：AIME数学题
- 固定：学习率1e-6，KL系数0.05

**结果对比**：

```
Batch=64：
Step   Accuracy  Entropy  Avg_Reward
100    0.32      1.65     0.08
200    0.41      1.42     0.15
300    0.35      1.58     0.11  ← 退步！
400    0.48      1.35     0.18
500    0.44      1.48     0.14  ← 又退步

Batch=512：
Step   Accuracy  Entropy  Avg_Reward
100    0.35      1.72     0.10
200    0.42      1.68     0.16
300    0.48      1.64     0.22  ← 持续进步
400    0.53      1.60     0.28
500    0.57      1.56     0.34
```

**关键观察**：
- 小批量：指标波动>10%
- 大批量：单调上升，波动<3%

### 2.3 显存受限时的解决方案：梯度累积

**问题**：单卡显存只能放128样本

**解决**：
```python
effective_batch_size = 512
micro_batch_size = 128
accumulation_steps = 512 // 128  # = 4

optimizer.zero_grad()
for i in range(accumulation_steps):
    micro_batch = get_batch(micro_batch_size)
    loss = compute_loss(micro_batch)

    # 关键：loss要除以累积步数
    (loss / accumulation_steps).backward()

# 累积4次后才更新
optimizer.step()
```

**等价性证明**：
```
大批量梯度：
∇ = (1/512) * Σ[i=1..512] ∇L_i

梯度累积：
∇ = (1/128)*Σ[1..128]∇L_i + (1/128)*Σ[129..256]∇L_i + ...
  = (1/512) * Σ[i=1..512] ∇L_i  ← 完全相同！
```

---

## 三、超参数调优：固定vs动态

### 3.1 JustRL的极简策略：全程固定

**固定的超参数**：
```python
config = {
    "learning_rate": 1e-6,       # 不使用学习率衰减
    "kl_coef": 0.05,             # 不动态调整
    "clip_epsilon": 0.2,         # PPO标准值
    "value_loss_coef": 0.5,      # 价值函数损失权重
    "entropy_coef": 0.01,        # 熵正则化系数
}

# 训练4000步，这些值一次都不改！
```

**为什么固定可行？**
1. **大批量消除了震荡**：不需要动态调整来"救火"
2. **足够的训练步数**：4000步足够让策略平滑收敛
3. **简化调试**：不稳定时容易定位问题

### 3.2 传统方法的动态调整（对比）

**常见做法**：
```python
# 学习率衰减
lr = initial_lr * (0.9 ** (step // 500))

# KL系数自适应
if kl_div > target_kl * 1.5:
    kl_coef *= 2  # KL太大，增强惩罚
elif kl_div < target_kl * 0.5:
    kl_coef *= 0.5  # KL太小，放松约束
```

**问题**：
- 引入新的超参数（衰减率、目标KL阈值）
- 难以复现（依赖训练动态）
- 掩盖根本问题（小批量的高方差）

### 3.3 实验对比：固定vs动态

**测试配置**：
- 方法A：固定超参 + Batch=512
- 方法B：动态LR调整 + Batch=128
- 方法C：动态KL调整 + Batch=128

**结果**：

| 指标 | 方法A（固定） | 方法B（动态LR） | 方法C（动态KL） |
|------|--------------|----------------|----------------|
| 最终准确率 | **58.6%** | 52.3% | 54.1% |
| 训练稳定性 | 平滑 | 中度波动 | 轻度波动 |
| 调试难度 | 低 | 高 | 中 |
| 可复现性 | 强 | 弱 | 中 |

**结论**：在大批量下，简单固定超参效果最好。

---

## 四、场景实战：从震荡到稳定的完整案例

### 4.1 初始问题：小批量训练崩溃

**场景描述**：
- 任务：训练1.5B模型解数学题
- 配置：Batch=64, LR=1e-6, KL=0.05
- 现象：训练500步后奖励骤降

**调试过程**：

**第1步：监控关键指标**
```python
class StabilityMonitor:
    def __init__(self):
        self.alerts = []

    def check(self, metrics, step):
        # 检测1：策略熵崩溃
        if metrics["entropy"] < 0.5:
            self.alerts.append(f"Step {step}: 策略熵过低 {metrics['entropy']:.2f}")

        # 检测2：KL散度爆炸
        if metrics["kl_div"] > 0.5:
            self.alerts.append(f"Step {step}: KL散度过大 {metrics['kl_div']:.2f}")

        # 检测3：奖励下降
        if len(metrics["reward_history"]) > 10:
            recent = metrics["reward_history"][-10:]
            if max(recent) - min(recent) > 0.3:
                self.alerts.append(f"Step {step}: 奖励剧烈波动")

        return self.alerts
```

**发现问题**：
```
Step 450: KL散度过大 0.62
Step 460: 奖励剧烈波动 (0.25→0.08)
Step 470: 策略熵过低 0.42
```

**第2步：诊断根因**
```python
# 检查批量梯度方差
def compute_gradient_variance(model, rollouts, num_samples=10):
    gradients = []

    for _ in range(num_samples):
        # 随机采样一个小批量
        batch = random.sample(rollouts, 64)
        loss = compute_loss(model, batch)

        # 记录梯度
        grads = torch.autograd.grad(loss, model.parameters())
        gradients.append([g.clone() for g in grads])

    # 计算方差
    variances = []
    for i in range(len(gradients[0])):
        param_grads = torch.stack([g[i].flatten() for g in gradients])
        variance = param_grads.var(dim=0).mean().item()
        variances.append(variance)

    return np.mean(variances)

# 结果
variance_64 = 0.082   # 高方差
variance_512 = 0.014  # 降低6倍！
```

**第3步：应用解决方案**
```python
# 方案1：增大批量（推荐）
config["batch_size"] = 512
config["gradient_accumulation"] = 8  # 如果显存不够

# 方案2（备选）：如果实在无法增大批量
config["learning_rate"] *= 0.5  # 减小学习率缓解
config["kl_coef"] *= 2          # 增强约束
```

### 4.2 优化后的训练曲线

**对比可视化**：

```
优化前（Batch=64）:
Reward │     ╱╲    ╱╲
       │    ╱  ╲  ╱  ╲╱
       │   ╱    ╲╱
       └───────────────────> Step

优化后（Batch=512）:
Reward │         ╱────
       │       ╱
       │     ╱
       │   ╱
       └───────────────────> Step
```

**数值对比**：
```
Batch=64:
- 最终奖励：0.18±0.12（均值±标准差）
- 崩溃次数：3次（需手动重启）
- 训练时长：8小时（含调试）

Batch=512:
- 最终奖励：0.38±0.02
- 崩溃次数：0次
- 训练时长：6小时（一次成功）
```

---

## 五、进阶技巧：精细化稳定性控制

### 5.1 自适应KL目标（保守做法）

**适用场景**：无法增大批量，且训练仍不稳定

**实现**：
```python
class AdaptiveKLController:
    def __init__(self, target_kl=0.01, horizon=200):
        self.target_kl = target_kl
        self.horizon = horizon
        self.kl_coef = 0.05

    def update(self, kl_div):
        """根据实际KL调整系数"""
        if kl_div > self.target_kl * 2:
            # KL太大，增强约束
            self.kl_coef = min(self.kl_coef * 1.5, 1.0)
        elif kl_div < self.target_kl * 0.5:
            # KL太小，放松约束
            self.kl_coef = max(self.kl_coef * 0.8, 0.01)

        return self.kl_coef

# 使用
kl_controller = AdaptiveKLController()
for step in range(4000):
    rollouts = sample_rollouts(model, problems)
    kl_div = compute_kl(rollouts)

    current_kl_coef = kl_controller.update(kl_div)
    loss = policy_loss + current_kl_coef * kl_div

    loss.backward()
    optimizer.step()
```

**效果**：
- ✅ 防止策略偏离太远
- ❌ 引入新超参数（target_kl, 调整速率）
- ⚠️ 仅在批量<256时考虑

### 5.2 梯度裁剪（标配技术）

**原理**：限制梯度范数，防止单步更新过大

```python
# 在optimizer.step()之前
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # 推荐值：0.5-2.0
)
```

**实验数据**：
| max_norm | 策略崩溃率 | 收敛速度 |
|----------|-----------|---------|
| 无裁剪 | 35% | 快（如果不崩） |
| 2.0 | 12% | 中等 |
| 1.0 | 3% | 稍慢但稳定 |
| 0.5 | 0% | 慢 |

**推荐**：总是使用1.0，成本可忽略。

### 5.3 价值函数稳定化

**问题**：PPO需要估计价值函数V(s)来计算优势A(s,a)

**常见不稳定**：
```python
# 价值函数损失
value_loss = (V_pred - V_target)^2

# 如果V_target波动大，价值网络学习不稳定
```

**解决方案1：裁剪价值损失**
```python
def compute_value_loss_clipped(v_pred, v_old, v_target, clip_epsilon=0.2):
    """PPO原论文推荐"""
    # 未裁剪损失
    loss_unclipped = (v_pred - v_target) ** 2

    # 裁剪预测值
    v_pred_clipped = v_old + torch.clamp(
        v_pred - v_old,
        -clip_epsilon,
        clip_epsilon
    )
    loss_clipped = (v_pred_clipped - v_target) ** 2

    # 取最大值（保守更新）
    return torch.max(loss_unclipped, loss_clipped).mean()
```

**解决方案2：增大价值损失权重**
```python
total_loss = policy_loss + 1.0 * value_loss  # 默认0.5，增大到1.0
```

**效果对比**：
- 无裁剪：价值估计RMSE=0.25
- 裁剪+权重1.0：RMSE=0.12（降低52%）

---

## 六、完整训练脚本：稳定性最佳实践

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

class StableRLTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"]
        )

        # 稳定性组件
        self.gradient_clipper = lambda: torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

    def train(self, train_data, num_steps=4000):
        """主训练循环"""
        metrics_history = {
            "reward": [],
            "entropy": [],
            "kl_div": [],
            "value_loss": []
        }

        for step in range(num_steps):
            # 1. 采样（使用大批量）
            rollouts = self.sample_rollouts(
                train_data,
                batch_size=self.config["batch_size"]
            )

            # 2. 计算优势
            rollouts = self.compute_advantages(rollouts)

            # 3. 多轮PPO更新
            for epoch in range(4):
                losses = self.ppo_update(rollouts)

            # 4. 记录指标
            metrics = self.compute_metrics(rollouts)
            for key in metrics_history:
                metrics_history[key].append(metrics[key])

            # 5. 稳定性检查
            if step % 100 == 0:
                self.stability_check(metrics_history, step)

            # 6. 定期评估
            if step % 500 == 0:
                val_acc = self.evaluate(val_data)
                print(f"Step {step}: Val Accuracy = {val_acc:.2%}")

        return metrics_history

    def ppo_update(self, rollouts):
        """单轮PPO更新（包含所有稳定性技巧）"""
        dataloader = DataLoader(
            rollouts,
            batch_size=self.config["micro_batch_size"],
            shuffle=True
        )

        accumulation_steps = self.config["batch_size"] // self.config["micro_batch_size"]
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # 前向传播
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 计算损失
            policy_loss = self.compute_policy_loss(batch, outputs)
            value_loss = self.compute_value_loss_clipped(batch, outputs)
            entropy_bonus = self.compute_entropy(outputs.logits)

            loss = (
                policy_loss +
                self.config["value_loss_coef"] * value_loss -
                self.config["entropy_coef"] * entropy_bonus
            )

            # 反向传播（归一化）
            (loss / accumulation_steps).backward()

            # 每N步更新一次
            if (i + 1) % accumulation_steps == 0:
                self.gradient_clipper()  # 梯度裁剪
                self.optimizer.step()
                self.optimizer.zero_grad()

    def stability_check(self, history, step):
        """稳定性监控"""
        # 检测1：奖励下降
        if len(history["reward"]) > 10:
            recent = history["reward"][-10:]
            if recent[-1] < min(recent[:-1]):
                print(f"⚠️  Step {step}: 奖励下降 {recent[-1]:.3f}")

        # 检测2：熵崩溃
        if history["entropy"][-1] < 0.5:
            print(f"⚠️  Step {step}: 策略熵过低 {history['entropy'][-1]:.2f}")

        # 检测3：KL爆炸
        if history["kl_div"][-1] > 0.5:
            print(f"⚠️  Step {step}: KL散度过大 {history['kl_div'][-1]:.3f}")
            print("   建议：增大kl_coef或减小learning_rate")

    def compute_value_loss_clipped(self, batch, outputs):
        """裁剪价值损失（PPO-Clip变体）"""
        v_pred = outputs.value  # 模型预测的价值
        v_old = batch["old_values"]  # 采样时的价值
        v_target = batch["returns"]  # 实际回报

        # 未裁剪损失
        loss_unclipped = (v_pred - v_target) ** 2

        # 裁剪预测值
        v_pred_clipped = v_old + torch.clamp(
            v_pred - v_old,
            -self.config["clip_epsilon"],
            self.config["clip_epsilon"]
        )
        loss_clipped = (v_pred_clipped - v_target) ** 2

        return torch.max(loss_unclipped, loss_clipped).mean()

# 使用示例
config = {
    "learning_rate": 1e-6,
    "batch_size": 512,
    "micro_batch_size": 64,  # 显存限制
    "kl_coef": 0.05,
    "clip_epsilon": 0.2,
    "value_loss_coef": 1.0,  # 增强价值学习
    "entropy_coef": 0.01,
}

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-1.5b")
trainer = StableRLTrainer(model, config)
history = trainer.train(train_data, num_steps=4000)
```

---

## 七、调试检查清单

### 训练前（0步）

- [ ] 批量大小>=256（显存允许尽量大）
- [ ] 梯度累积正确实现（loss要除以步数）
- [ ] 梯度裁剪已启用（max_norm=1.0）
- [ ] 采样温度>0（保持探索）

### 训练初期（0-500步）

- [ ] 策略熵在1.5-2.0之间
- [ ] 奖励曲线向上（允许小波动）
- [ ] KL散度<0.1
- [ ] 价值函数损失下降

### 训练中期（500-2000步）

- [ ] 奖励持续上升（斜率可变缓）
- [ ] 策略熵平滑下降到1.0-1.5
- [ ] 无突然的指标跳变
- [ ] 验证集性能同步提升

### 训练后期（2000-4000步）

- [ ] 三条曲线趋于平稳
- [ ] 策略熵稳定在0.8-1.2
- [ ] 验证集不再提升（可早停）
- [ ] KL散度仍<0.1

---

## 八、常见问题诊断表

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| **奖励不涨** | 学习率太小 | 增大10倍试试 |
| | 批量太小 | 增大到512+ |
| | 奖励函数错误 | 检查验证逻辑 |
| **奖励突然跌** | 学习率太大 | 减小10倍 |
| | KL系数太小 | 增大到0.1 |
| | 没有梯度裁剪 | 添加clip_grad_norm |
| **熵快速崩溃** | 温度太低 | 采样时temperature=0.8 |
| | 熵惩罚太弱 | entropy_coef增大到0.02 |
| | 长度惩罚过强 | 移除显式惩罚 |
| **KL散度爆炸** | 裁剪阈值太大 | clip_epsilon降到0.1 |
| | 更新轮数太多 | PPO epoch从4降到2 |
| | 批量太小 | 增大批量 |
| **显存溢出** | 批量设置错误 | 检查梯度累积实现 |
| | 序列太长 | 截断到2048 |
| | 没用混合精度 | 启用torch.cuda.amp |

---

## 九、总结与最佳实践

### 核心原则

1. **批量优先**：稳定性问题首先尝试增大批量
2. **固定超参**：在大批量下，固定超参效果更好
3. **监控到位**：实时跟踪熵/奖励/KL三大指标
4. **保守更新**：梯度裁剪+价值损失裁剪是标配

### 推荐配置模板

```python
# 小模型（1.5B-3B）
config_small = {
    "learning_rate": 1e-6,
    "batch_size": 512,
    "kl_coef": 0.05,
    "clip_epsilon": 0.2,
    "gradient_clip": 1.0,
}

# 大模型（7B-13B）
config_large = {
    "learning_rate": 5e-7,      # 减半
    "batch_size": 256,          # 显存受限
    "gradient_accumulation": 16, # 模拟大batch
    "kl_coef": 0.1,             # 更强约束
    "clip_epsilon": 0.2,
    "gradient_clip": 0.5,       # 更保守
}
```

### 下一步学习

- [奖励函数设计完全指南](08_奖励信号设计_从稀疏到稠密的工程实践.md)
- [模型优化反模式：哪些技巧可能有害](09_LLM优化反模式_复杂技巧的负面案例.md)
