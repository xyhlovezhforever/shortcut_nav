# 参数高效微调（PEFT）技术详解

## 一、概述

参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）是在资源受限情况下微调大语言模型的关键技术。通过只训练少量参数，PEFT能够在保持模型性能的同时，大幅降低计算和存储成本。

---

## 二、为什么需要PEFT

### 2.1 全参数微调的挑战

**问题**：
1. **内存占用巨大**：需要存储梯度、优化器状态（Adam需要2倍参数量）
2. **计算成本高**：需要更新所有参数
3. **存储成本**：每个任务需要保存一个完整模型副本
4. **灾难性遗忘**：可能破坏预训练知识

**示例**：
```
LLaMA-7B全参数微调：
- 模型参数：7B × 4字节 = 28GB
- 梯度：7B × 4字节 = 28GB
- Adam状态：7B × 8字节 = 56GB
总计：~112GB（单个GPU装不下）
```

### 2.2 PEFT的优势

| 特性 | 全参数微调 | PEFT |
|-----|-----------|------|
| 可训练参数 | 100% | 0.1%-10% |
| GPU内存 | 100GB+ | 10-30GB |
| 训练时间 | 长 | 短 |
| 存储成本 | 高（每任务一个完整模型） | 低（只存储小适配器） |
| 多任务服务 | 困难 | 容易（动态加载适配器） |

---

## 三、主流PEFT方法

### 3.1 LoRA（Low-Rank Adaptation）

#### 3.1.1 核心原理

**论文**：《LoRA: Low-Rank Adaptation of Large Language Models》（Microsoft, 2021）

**核心思想**：
- 冻结预训练权重
- 在每层添加低秩矩阵分解
- 只训练低秩矩阵

**数学表示**：
```
原始前向传播：h = W₀x

LoRA前向传播：h = W₀x + ΔWx = W₀x + BAx

其中：
- W₀ ∈ R^(d×k)：冻结的预训练权重
- B ∈ R^(d×r), A ∈ R^(r×k)：可训练的低秩矩阵
- r << min(d, k)：秩（通常r=8,16,32）
```

**参数量对比**：
```
原始：d × k
LoRA：d × r + r × k ≈ 2dr（当r << d,k时）

例如：d=4096, k=4096, r=8
原始：16,777,216参数
LoRA：65,536参数（压缩256倍）
```

#### 3.1.2 实现示例

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x, pretrained_output):
        # 原始输出 + LoRA增量
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return pretrained_output + lora_output

# 使用示例
class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=8):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False  # 冻结原始权重

        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank
        )

    def forward(self, x):
        pretrained_out = self.linear(x)
        return self.lora(x, pretrained_out)
```

#### 3.1.3 使用Hugging Face PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 秩
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.1,       # Dropout率
    target_modules=["q_proj", "v_proj"],  # 应用LoRA的模块
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

#### 3.1.4 LoRA的变体

**1. AdaLoRA（Adaptive LoRA）**
- 自适应调整每层的秩
- 重要层使用更高的秩
- 进一步减少参数

**2. LoRA+**
- 改进的初始化策略
- 更好的收敛性能

**3. DoRA（Weight-Decomposed LoRA）**
- 分解权重的幅度和方向
- 更好的表达能力

---

### 3.2 QLoRA（Quantized LoRA）

#### 3.2.1 核心创新

**论文**：《QLoRA: Efficient Finetuning of Quantized LLMs》（University of Washington, 2023）

**核心思想**：
- 将基础模型量化到4-bit
- 在量化模型上应用LoRA
- 使用特殊的数据类型（NF4）

**内存节省**：
```
LLaMA-7B：
- FP16全参数：~14GB
- FP16 + LoRA：~16GB（需要存储梯度）
- 4-bit + QLoRA：~5GB（减少70%内存）

可在单张RTX 3090（24GB）上微调LLaMA-13B
可在单张A100（40GB）上微调LLaMA-65B
```

#### 3.2.2 关键技术

**1. NF4（4-bit NormalFloat）**
- 专为正态分布权重设计的数据类型
- 更好地保留模型性能

**2. 双重量化（Double Quantization）**
- 量化量化常数本身
- 进一步节省内存

**3. 分页优化器（Paged Optimizers）**
- 使用CPU内存缓存
- 避免OOM错误

#### 3.2.3 实现示例

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用FP16
    bnb_4bit_use_double_quant=True,      # 双重量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"  # 自动分配到多个GPU
)

# 准备模型
model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 训练
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---

### 3.3 Prefix Tuning

#### 3.3.1 原理

**核心思想**：
- 在输入序列前添加可训练的"虚拟tokens"
- 只优化这些prefix的embeddings
- 冻结模型主体

**架构**：
```
输入：[PREFIX] [ACTUAL_INPUT]
      ↑可训练   ↑冻结

示例：
PREFIX: [P₁, P₂, ..., P_k]（可训练）
输入：  "Translate to French: Hello"
```

#### 3.3.2 实现

```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # Prefix长度
    encoder_hidden_size=128,
)

model = get_peft_model(model, config)
```

**优缺点**：
- 优点：参数更少
- 缺点：推理时增加序列长度

---

### 3.4 Prompt Tuning

#### 3.4.1 原理

**简化版Prefix Tuning**：
- 只在输入层添加soft prompts
- 不影响中间层

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",  # 或 "RANDOM"
    prompt_tuning_init_text="Classify if the tweet is positive or negative:",
    tokenizer_name_or_path="gpt2",
)

model = get_peft_model(model, config)
```

---

### 3.5 Adapter Tuning

#### 3.5.1 原理

**核心思想**：
- 在Transformer每层插入小型adapter模块
- 只训练adapter，冻结主模型

**架构**：
```
输入
  ↓
LayerNorm
  ↓
Self-Attention（冻结）
  ↓
Adapter（可训练）← 瓶颈结构
  ↓
LayerNorm
  ↓
Feed-Forward（冻结）
  ↓
Adapter（可训练）
  ↓
输出
```

#### 3.5.2 Adapter结构

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        # 残差连接
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual
```

---

### 3.6 IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）

#### 3.6.1 原理

**核心思想**：
- 通过可学习的向量缩放激活
- 参数极少（< 0.01%）

```python
# IA³ 在attention和FFN中缩放激活
class IA3Layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 可学习的缩放向量
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.scale
```

---

## 四、PEFT方法对比

| 方法 | 参数量 | 训练速度 | 推理速度 | 性能 | 适用场景 |
|-----|-------|---------|---------|------|---------|
| LoRA | ~0.1-1% | 快 | 无损 | 优秀 | **通用推荐** |
| QLoRA | ~0.1-1% | 快 | 无损 | 优秀 | **GPU受限** |
| Prefix Tuning | ~0.01-0.1% | 快 | 稍慢 | 良好 | 小任务 |
| Adapter | ~1-5% | 中等 | 稍慢 | 优秀 | 多任务 |
| IA³ | < 0.01% | 很快 | 无损 | 良好 | 极低资源 |
| 全参数 | 100% | 慢 | 标准 | 最优 | 资源充足 |

---

## 五、实战案例

### 5.1 消费级GPU微调LLaMA-7B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# 1. 加载量化模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 2. 准备模型并配置LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 3. 加载数据集
dataset = load_dataset("timdettmers/openassistant-guanaco")

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./qlora-llama2-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效batch_size=16
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_8bit",  # 分页优化器
)

# 5. 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# 6. 保存LoRA权重（只有几MB）
model.save_pretrained("./lora-adapter")
```

---

### 5.2 多任务适配器管理

```python
from peft import PeftModel

# 基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载不同任务的适配器
model_task1 = PeftModel.from_pretrained(base_model, "./lora-translation")
model_task2 = PeftModel.from_pretrained(base_model, "./lora-summarization")

# 动态切换
def inference(text, task="translation"):
    if task == "translation":
        model = model_task1
    else:
        model = model_task2

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0])
```

---

### 5.3 合并LoRA权重（部署优化）

```python
from peft import PeftModel

# 加载基础模型和LoRA
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
lora_model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 合并权重
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型（推理时不需要PEFT库）
merged_model.save_pretrained("./merged-model")

# 推理（使用标准transformers）
from transformers import pipeline
pipe = pipeline("text-generation", model="./merged-model")
```

---

## 六、最佳实践

### 6.1 选择合适的方法

**决策树**：
```
GPU内存充足（> 40GB）？
├─ 是 → 考虑全参数微调或LoRA（r=64）
└─ 否 → GPU内存 > 16GB？
    ├─ 是 → LoRA（r=8-16）
    └─ 否 → QLoRA（4-bit + r=8）

数据量很小（< 1000样本）？
└─ 是 → Prompt Tuning或IA³

需要服务多个任务？
└─ 是 → Adapter或LoRA（易于切换）
```

### 6.2 超参数调优

**LoRA关键参数**：
```python
# 1. 秩（r）
r=4    # 轻量任务（情感分类）
r=8    # 常规任务（推荐）
r=16   # 复杂任务（代码生成）
r=64   # 接近全参数性能

# 2. alpha（通常设为r的2倍）
lora_alpha = 2 * r

# 3. target_modules（应用LoRA的层）
# 最小配置
target_modules=["q_proj", "v_proj"]

# 推荐配置
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# 完整配置（包括FFN）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

### 6.3 常见问题

**Q1: LoRA性能不如全参数？**
```
解决方案：
1. 增加秩（r=8 → r=16）
2. 扩大target_modules范围
3. 调整学习率（LoRA通常需要更高的lr）
4. 增加训练步数
```

**Q2: QLoRA训练不稳定？**
```
解决方案：
1. 使用梯度裁剪（max_grad_norm=0.3）
2. 降低学习率
3. 增加warmup_steps
4. 使用bf16而非fp16（如果硬件支持）
```

**Q3: 内存仍然不够？**
```
解决方案：
1. 减小batch_size，增加gradient_accumulation_steps
2. 使用8-bit优化器（paged_adamw_8bit）
3. 启用gradient_checkpointing
4. 使用DeepSpeed ZeRO
```

---

## 七、未来趋势

### 7.1 新兴技术

1. **Delta Tuning**：统一的参数高效框架
2. **MoE-LoRA**：混合专家LoRA
3. **Adapter Fusion**：融合多个适配器
4. **AutoPEFT**：自动搜索最优PEFT配置

### 7.2 工业应用

- **多租户服务**：一个基础模型 + 多个租户适配器
- **边缘部署**：量化模型 + 小适配器
- **持续学习**：增量添加适配器，避免遗忘

---

## 八、总结

PEFT技术使得在资源受限环境下微调大模型成为可能：

1. **推荐组合**：QLoRA（4-bit）是性价比最高的方案
2. **参数选择**：r=8-16适合大多数任务
3. **部署策略**：训练时用LoRA，部署时合并权重
4. **成本对比**：QLoRA可减少70%内存，同时保持95%+性能

**关键要点**：
- LoRA：通用首选
- QLoRA：GPU受限首选
- Adapter：多任务场景
- 量化+PEFT：最大化资源利用

通过合理使用PEFT技术，即使是个人开发者也能微调70B级别的模型！
