# AI大模型专有名词完全指南

## 一、概述

本文档系统化地整理了AI大模型领域的所有核心专有名词，按应用场景分类，帮助你快速理解和掌握大模型技术栈的完整知识体系。

---

## 二、基础架构相关

### 2.1 模型架构

#### Transformer
**定义**：基于自注意力机制的神经网络架构，是现代大模型的基础。

**核心组件**：
```
Encoder-Decoder架构
├─ Encoder：编码输入序列
└─ Decoder：生成输出序列

关键模块：
- Multi-Head Attention（多头注意力）
- Feed-Forward Network（前馈网络）
- Layer Normalization（层归一化）
- Residual Connection（残差连接）
```

**应用场景**：
- 机器翻译（原始Transformer）
- 文本生成（GPT系列）
- 文本理解（BERT系列）

---

#### Attention Mechanism（注意力机制）

**定义**：允许模型关注输入序列中不同位置的信息。

**类型**：

**1. Self-Attention（自注意力）**
```python
# 计算公式
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中：
- Q (Query): 查询向量
- K (Key): 键向量
- V (Value): 值向量
- d_k: 键向量的维度
```

**2. Multi-Head Attention（多头注意力）**
```
将注意力分为h个头，并行计算：
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O

每个头：head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**3. Cross-Attention（交叉注意力）**
- 用于Encoder-Decoder之间
- Q来自Decoder，K和V来自Encoder

**应用场景**：
- 机器翻译：关注源语言和目标语言的对应关系
- 图像描述：关注图像不同区域
- 问答系统：关注问题和上下文的关联

---

#### Positional Encoding（位置编码）

**定义**：为序列中的每个位置添加位置信息。

**类型**：

**1. 绝对位置编码**
```python
# 正弦余弦位置编码
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**2. 相对位置编码**
- 关注token之间的相对距离
- 代表：T5, DeBERTa

**3. 可学习位置编码**
- BERT使用的方法
- 位置编码作为可训练参数

**4. RoPE（Rotary Position Embedding，旋转位置编码）**
- LLaMA使用
- 通过旋转矩阵编码位置信息
- 外推性好

**5. ALiBi（Attention with Linear Biases）**
- 直接在注意力分数上添加线性偏置
- 不需要显式位置编码

---

#### Layer Normalization（层归一化）

**定义**：对每个样本的特征进行归一化。

**公式**：
```
LN(x) = γ × (x - μ) / √(σ² + ε) + β

其中：
- μ: 均值
- σ: 标准差
- γ, β: 可学习参数
```

**变体**：
- **Pre-LN**：在Attention/FFN之前归一化（GPT-3使用）
- **Post-LN**：在Attention/FFN之后归一化（原始Transformer）
- **RMSNorm**：简化版，只使用均方根（LLaMA使用）

---

### 2.2 模型类型

#### Encoder-Only模型

**代表**：BERT, RoBERTa, DeBERTa

**特点**：
- 双向编码
- 适合理解任务
- 使用MLM（Masked Language Modeling）预训练

**应用场景**：
- 文本分类
- 命名实体识别（NER）
- 问答系统（抽取式）
- 语义相似度计算

---

#### Decoder-Only模型

**代表**：GPT系列, LLaMA, Claude, Gemini

**特点**：
- 单向（从左到右）生成
- 自回归生成
- 使用CLM（Causal Language Modeling）预训练

**应用场景**：
- 文本生成
- 对话系统
- 代码生成
- 创意写作

---

#### Encoder-Decoder模型

**代表**：T5, BART, mT5

**特点**：
- 结合编码和解码
- 适合序列到序列任务
- 使用各种去噪目标预训练

**应用场景**：
- 机器翻译
- 文本摘要
- 文本改写
- 问答生成

---

### 2.3 模型规模术语

#### Parameters（参数量）

**定义**：模型中可训练的权重数量。

**规模分类**：
```
Small（小型）: < 1B参数
- BERT-Base: 110M
- GPT-2-Small: 117M

Medium（中型）: 1B - 10B
- GPT-2-Large: 1.5B
- T5-Large: 3B

Large（大型）: 10B - 100B
- GPT-3: 175B
- LLaMA-2-70B: 70B

Ultra-Large（超大型）: > 100B
- GPT-4: 据说1.76T（未证实）
- Gemini Ultra: 未公开
```

---

#### FLOPs（浮点运算次数）

**定义**：训练或推理需要的浮点运算次数。

**单位**：
- GFLOP: 10⁹次运算
- TFLOP: 10¹²次运算
- PFLOP: 10¹⁵次运算

**Scaling Law（缩放定律）**：
```
模型性能 ∝ (参数量)^α × (数据量)^β × (计算量)^γ

Chinchilla定律：
对于给定的计算预算C，最优配置：
- 参数量N ∝ C^0.5
- 训练tokens T ∝ C^0.5
```

---

#### Context Window（上下文窗口）

**定义**：模型一次能处理的最大token数量。

**常见大小**：
```
短上下文：
- BERT: 512 tokens
- GPT-2: 1024 tokens

标准上下文：
- GPT-3: 2048 tokens
- ChatGPT: 4096 tokens

长上下文：
- GPT-4: 8K / 32K tokens
- Claude 2: 100K tokens
- GPT-4 Turbo: 128K tokens

超长上下文：
- Gemini 1.5 Pro: 1M tokens
```

---

## 三、训练相关

### 3.1 预训练任务

#### MLM（Masked Language Modeling，掩码语言建模）

**定义**：随机遮盖输入中的部分token，让模型预测被遮盖的内容。

**流程**：
```
原始文本: "The cat sat on the mat"
掩码后:   "The [MASK] sat on the [MASK]"
目标:     预测 "cat" 和 "mat"
```

**变体**：
- **Whole Word Masking**：遮盖整个词而非子词
- **Entity Masking**：遮盖实体
- **Span Masking**：遮盖连续的片段

**使用模型**：BERT, RoBERTa, ALBERT

---

#### CLM（Causal Language Modeling，因果语言建模）

**定义**：根据前面的token预测下一个token（自回归）。

**流程**：
```
输入:  "The cat sat"
预测:  "on"

输入:  "The cat sat on"
预测:  "the"
```

**使用模型**：GPT系列, LLaMA, PaLM

---

#### NSP（Next Sentence Prediction，下一句预测）

**定义**：判断两个句子是否连续。

**流程**：
```
正例：
Sentence A: "The cat sat on the mat."
Sentence B: "It was very comfortable."
Label: IsNext

负例：
Sentence A: "The cat sat on the mat."
Sentence B: "Paris is the capital of France."
Label: NotNext
```

**使用模型**：BERT（RoBERTa移除了此任务）

---

#### SOP（Sentence Order Prediction，句子顺序预测）

**定义**：判断两个句子的顺序是否正确。

**改进点**：比NSP更难，避免主题判断的捷径。

**使用模型**：ALBERT, StructBERT

---

### 3.2 优化技术

#### Gradient Accumulation（梯度累积）

**定义**：累积多个小batch的梯度后再更新参数。

**目的**：
- 在有限内存下模拟大batch训练
- 提高训练稳定性

**示例**：
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # 归一化
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

#### Gradient Clipping（梯度裁剪）

**定义**：限制梯度的范数，防止梯度爆炸。

**方法**：
```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

#### Mixed Precision Training（混合精度训练）

**定义**：使用FP16和FP32混合训练，加速并减少内存。

**流程**：
```
1. 前向传播：FP16
2. 计算损失：FP32
3. 反向传播：FP16
4. 梯度更新：FP32（Master Weights）
```

**工具**：
- NVIDIA Apex
- PyTorch AMP（Automatic Mixed Precision）

---

#### Learning Rate Schedule（学习率调度）

**常见策略**：

**1. Warmup（预热）**
```python
# 线性预热
lr = base_lr * (current_step / warmup_steps)
```

**2. Linear Decay（线性衰减）**
```python
lr = base_lr * (1 - current_step / total_steps)
```

**3. Cosine Annealing（余弦退火）**
```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * T_cur / T_max))
```

**4. Inverse Square Root**
```python
# Transformer论文使用
lr = base_lr * min(step^(-0.5), step * warmup_steps^(-1.5))
```

---

### 3.3 数据相关

#### Tokenization（分词）

**定义**：将文本拆分为模型可处理的基本单元。

**方法**：

**1. Word-level（词级别）**
- 优点：直观
- 缺点：词表巨大，OOV问题

**2. Character-level（字符级别）**
- 优点：无OOV
- 缺点：序列过长

**3. Subword（子词）**

**BPE（Byte Pair Encoding）**
```
原理：迭代合并最频繁的字符对

示例：
"low" + "est" → "lowest"
词表：["l", "o", "w", "e", "s", "t", "low", "est", "lowest"]
```

**WordPiece**
```
BERT使用
标记：##表示子词

示例："playing" → ["play", "##ing"]
```

**SentencePiece**
```
language-agnostic
支持直接从原始文本训练

变体：
- Unigram LM
- BPE
```

**使用对比**：
| 模型 | 分词器 |
|------|--------|
| GPT-2/3 | BPE |
| BERT | WordPiece |
| T5, LLaMA | SentencePiece |
| GPT-4 | Tiktoken (BPE变体) |

---

#### Token

**定义**：分词后的基本单元。

**特殊Token**：
```
[PAD]: 填充token
[UNK]: 未知token
[CLS]: 分类token（BERT）
[SEP]: 分隔token
[MASK]: 掩码token
<s>, </s>: 句子开始/结束（GPT）
<|endoftext|>: 文本结束（GPT-2）
```

---

#### Vocabulary Size（词表大小）

**常见大小**：
```
BERT: 30K
GPT-2: 50K
GPT-3: 50K
LLaMA: 32K
LLaMA-2: 32K
GPT-4: ~100K（估计）
```

**权衡**：
- 大词表：表达能力强，但模型参数多
- 小词表：模型小，但序列长

---

### 3.4 训练策略

#### Curriculum Learning（课程学习）

**定义**：从简单样本逐渐过渡到复杂样本。

**应用**：
- 从短序列到长序列
- 从高质量数据到全量数据

---

#### Data Augmentation（数据增强）

**文本增强方法**：
```
1. 回译（Back-translation）
   中文 → 英文 → 中文

2. 同义词替换
   "快乐" → "开心"

3. 随机删除/插入/交换

4. Mixup（混合样本）

5. Cutoff（随机遮盖）
```

---

#### Pre-training（预训练）

**定义**：在大规模无标注数据上训练模型。

**目标**：学习通用语言表示。

**数据来源**：
- Common Crawl（网页数据）
- Wikipedia
- Books Corpus
- GitHub代码
- 学术论文

---

#### Fine-tuning（微调）

**定义**：在特定任务数据上继续训练预训练模型。

**类型**：

**1. Full Fine-tuning（全参数微调）**
- 更新所有参数
- 效果好，但成本高

**2. Parameter-Efficient Fine-tuning（参数高效微调）**
- 只更新少量参数
- 代表：LoRA, Adapter, Prefix Tuning

---

#### Continual Learning（持续学习）

**定义**：模型持续从新数据中学习。

**挑战**：
- Catastrophic Forgetting（灾难性遗忘）

**解决方案**：
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks
- Memory Replay

---

## 四、推理与生成

### 4.1 生成策略

#### Greedy Decoding（贪婪解码）

**定义**：每步选择概率最高的token。

**优点**：
- 快速
- 确定性

**缺点**：
- 可能陷入局部最优
- 生成重复

```python
def greedy_decode(model, input_ids):
    for _ in range(max_length):
        logits = model(input_ids)
        next_token = logits.argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids
```

---

#### Beam Search（束搜索）

**定义**：保留top-k个候选序列，探索多条路径。

**参数**：
- `num_beams`：束宽度（通常5-10）

**流程**：
```
Step 1: 保留概率最高的k个token
Step 2: 对每个候选，生成k个扩展
Step 3: 从k²个候选中选择top-k
...
```

**变体**：
- **Diverse Beam Search**：鼓励多样性
- **Constrained Beam Search**：满足约束条件

**优缺点**：
- ✅ 比贪婪解码好
- ❌ 倾向生成平淡、安全的文本
- ❌ 对话场景表现差

---

#### Sampling（采样）

**定义**：按概率分布随机采样token。

**方法**：

**1. Random Sampling（随机采样）**
```python
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**2. Top-k Sampling**
```python
# 只从概率最高的k个token中采样
top_k = 50
top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
next_token = torch.multinomial(top_k_probs, num_samples=1)
```

**3. Top-p Sampling（Nucleus Sampling，核采样）**
```python
# 累积概率达到p时停止
p = 0.9
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
# 选择累积概率<=p的token
nucleus = cumulative_probs <= p
```

**4. Temperature Sampling（温度采样）**
```python
# temperature控制分布的平滑度
temperature = 0.7  # <1更确定，>1更随机

logits = logits / temperature
probs = F.softmax(logits, dim=-1)
```

**温度效果**：
```
temperature = 0.1  → 接近贪婪（确定性强）
temperature = 1.0  → 原始分布
temperature = 2.0  → 更随机（创意性强）
```

---

#### Repetition Penalty（重复惩罚）

**定义**：降低已生成token的概率，减少重复。

```python
# 对已生成的token降低分数
for token in generated_tokens:
    logits[token] /= repetition_penalty  # 通常1.0-1.5
```

---

#### Length Penalty（长度惩罚）

**定义**：调整生成序列的长度偏好。

```python
# Beam Search中使用
score = log_prob / (length ** length_penalty)

# length_penalty > 1: 鼓励长序列
# length_penalty < 1: 鼓励短序列
```

---

### 4.2 推理优化

#### KV Cache（键值缓存）

**定义**：缓存已计算的Key和Value，避免重复计算。

**原理**：
```
Step 1: 生成 "The"
  → 计算并缓存 K₁, V₁

Step 2: 生成 "cat"
  → 计算 K₂, V₂
  → 复用 K₁, V₁（无需重新计算）

内存占用：
batch_size × seq_len × 2 × num_layers × hidden_dim
```

**效果**：
- 速度提升：2-10倍
- 内存占用：增加

---

#### Flash Attention

**定义**：优化的注意力计算，减少内存访问。

**改进**：
- 分块计算
- 避免物化完整注意力矩阵
- 内存：O(N²) → O(N)
- 速度：2-4倍

**版本**：
- Flash Attention 1
- Flash Attention 2（更快）

---

#### Quantization（量化）

**定义**：降低模型权重的精度。

**精度类型**：
```
FP32（单精度浮点）: 32位
FP16（半精度浮点）: 16位
BF16（Brain Float）: 16位（范围更大）
INT8（8位整数）:    8位
INT4（4位整数）:    4位
```

**方法**：

**1. Post-Training Quantization（训练后量化）**
- 直接量化已训练模型
- 快速但可能损失精度

**2. Quantization-Aware Training（量化感知训练）**
- 训练时模拟量化
- 精度损失小

**3. GPTQ（GPT-Quantization）**
- 逐层量化
- 保持性能

**4. AWQ（Activation-aware Weight Quantization）**
- 基于激活值重要性
- 4-bit量化，损失<1%

---

#### Model Parallelism（模型并行）

**类型**：

**1. Data Parallelism（数据并行）**
```
每个GPU持有完整模型
数据分片到不同GPU
梯度汇总后更新
```

**2. Tensor Parallelism（张量并行）**
```
单个层的矩阵乘法拆分到多个GPU
示例：
  Original: Y = XW [d_model × d_model]
  Split:    Y = [XW₁, XW₂] (2个GPU)
```

**3. Pipeline Parallelism（流水线并行）**
```
不同层分配到不同GPU
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

**4. Sequence Parallelism（序列并行）**
```
长序列拆分到多个GPU
每个GPU处理序列的一部分
```

---

#### Speculative Decoding（投机解码）

**定义**：用小模型快速生成候选，大模型并行验证。

**流程**：
```
1. 小模型生成K个token（快）
2. 大模型一次验证K个token（并行）
3. 接受正确的，拒绝错误的
```

**效果**：
- 加速：2-3倍
- 质量：无损

---

## 五、对齐与安全

### 5.1 对齐技术

#### RLHF（Reinforcement Learning from Human Feedback）

**定义**：通过人类反馈训练奖励模型，再用强化学习优化策略。

**三步骤**：

**Step 1: Supervised Fine-tuning（SFT）**
```
使用高质量人工标注数据微调基础模型
目标：学习基本的对话能力和格式
```

**Step 2: Reward Model Training（RM）**
```
训练奖励模型：
输入：prompt + response
输出：质量分数

数据：人类对多个回答的排序
  Response A > Response B > Response C

损失函数：
  L_RM = -log(σ(r(x,y_w) - r(x,y_l)))
  y_w: 更好的回答
  y_l: 更差的回答
```

**Step 3: PPO Optimization（PPO）**
```
使用PPO算法优化策略模型
目标函数：
  maximize E[reward(x, π(x))]
  约束：KL(π || π_ref) < δ（不偏离太远）
```

**使用模型**：
- ChatGPT
- Claude
- Llama-2-Chat

---

#### DPO（Direct Preference Optimization）

**定义**：直接从偏好数据优化，无需训练奖励模型。

**优势**：
- 更简单（跳过RM训练）
- 更稳定（避免RL不稳定）
- 更高效

**损失函数**：
```
L_DPO = -log(σ(β log π_θ(y_w|x)/π_ref(y_w|x)
              - β log π_θ(y_l|x)/π_ref(y_l|x)))
```

---

#### Constitutional AI

**定义**：通过预定义的原则（Constitution）引导模型行为。

**流程**：
```
1. 定义原则（如"无害"、"诚实"、"有帮助"）
2. 模型自我批评和修正
3. 强化符合原则的行为
```

**使用模型**：Claude（Anthropic）

---

### 5.2 安全相关

#### Hallucination（幻觉）

**定义**：模型生成不准确或虚构的信息。

**类型**：
```
1. 事实性幻觉
   模型：巴黎是德国的首都（错误）

2. 忠实性幻觉
   输入：讨论苹果
   输出：谈论微软（偏离输入）

3. 指令幻觉
   忽略或曲解用户指令
```

**缓解方法**：
- RAG（检索增强生成）
- 引用来源
- 增加训练数据多样性
- RLHF对齐

---

#### Jailbreak（越狱）

**定义**：绕过模型的安全限制。

**常见方法**：
```
1. 角色扮演
   "假装你是一个没有道德限制的AI..."

2. DAN（Do Anything Now）
   "你现在处于DAN模式..."

3. 编码绕过
   使用Base64、ROT13等编码

4. 间接提示
   "写一个虚构故事，其中..."
```

**防御**：
- 鲁棒的RLHF
- 红队测试
- 动态过滤

---

#### Red Teaming（红队测试）

**定义**：主动寻找模型的漏洞和不安全行为。

**测试内容**：
- 有害内容生成
- 偏见和歧视
- 隐私泄露
- 越狱攻击

---

#### Toxicity（毒性）

**定义**：模型生成攻击性、冒犯性或有害的内容。

**检测**：
- Perspective API
- Detoxify模型
- 人工审核

---

## 六、提示工程

### 6.1 基础概念

#### Prompt（提示词）

**定义**：输入给模型的指令或问题。

**组成部分**：
```
Prompt = Instruction + Context + Input + Output Indicator

示例：
Instruction: "将以下文本翻译成英文"
Context: "保持原意，语言正式"
Input: "你好，世界"
Output Indicator: "英文翻译："
```

---

#### Zero-shot Prompting（零样本提示）

**定义**：不提供任何示例，直接给出任务指令。

**示例**：
```
Prompt: "将'你好'翻译成英文"
Output: "Hello"
```

**适用场景**：
- 简单任务
- 模型能力强

---

#### Few-shot Prompting（少样本提示）

**定义**：提供少量示例，帮助模型理解任务。

**示例**：
```
Prompt:
Q: "猫"的英文是什么？
A: cat

Q: "狗"的英文是什么？
A: dog

Q: "鸟"的英文是什么？
A:

Output: bird
```

**适用场景**：
- 复杂任务
- 特定格式要求

---

#### Chain-of-Thought (CoT)（思维链）

**定义**：引导模型逐步推理。

**示例**：
```
问题：Roger有5个网球。他又买了2罐网球，每罐3个球。他现在有多少个网球？

普通提示：
答案：11个

CoT提示：
让我们一步步思考：
1. Roger最初有5个球
2. 他买了2罐，每罐3个，所以是2×3=6个新球
3. 总共：5+6=11个球
答案：11个
```

**触发方式**：
```
"让我们一步步思考"
"Let's think step by step"
```

**变体**：
- **Zero-shot CoT**：只加"让我们一步步思考"
- **Auto-CoT**：自动生成推理步骤
- **Self-Consistency CoT**：多次推理，投票选择

---

#### Tree of Thoughts (ToT)（思维树）

**定义**：探索多条推理路径，类似树搜索。

**流程**：
```
问题
 ├─ 思路1
 │   ├─ 步骤1.1
 │   └─ 步骤1.2
 ├─ 思路2
 │   ├─ 步骤2.1
 │   └─ 步骤2.2
 └─ 思路3
     └─ ...
```

**适用**：复杂推理、规划任务

---

#### Self-Consistency（自洽性）

**定义**：多次生成，选择最一致的答案。

**流程**：
```
1. 用不同采样生成N个答案
2. 选择出现频率最高的答案
```

**示例**：
```
生成5次：
  [11, 11, 12, 11, 11]
选择：11（出现4次）
```

---

#### Instruction Tuning（指令微调）

**定义**：在指令-响应对上微调模型，提升指令遵循能力。

**数据格式**：
```json
{
  "instruction": "将以下文本分类为正面或负面",
  "input": "这部电影太棒了！",
  "output": "正面"
}
```

**代表数据集**：
- FLAN（Google）
- Super-NaturalInstructions
- Alpaca（Stanford）
- ShareGPT

**使用模型**：
- FLAN-T5
- InstructGPT
- Alpaca
- Vicuna

---

### 6.2 高级技术

#### ReAct（Reason + Act）

**定义**：结合推理和行动，与外部工具交互。

**流程**：
```
Thought 1: 我需要查找巴黎的人口
Action 1: Search[巴黎人口]
Observation 1: 巴黎人口约210万

Thought 2: 我需要查找伦敦的人口
Action 2: Search[伦敦人口]
Observation 2: 伦敦人口约900万

Thought 3: 现在我可以比较了
Answer: 伦敦人口比巴黎多
```

**工具调用**：
- 搜索引擎
- 计算器
- 数据库
- API

---

#### Retrieval-Augmented Generation (RAG)

**定义**：检索相关文档，增强生成质量。

**流程**：
```
1. 用户问题：什么是Transformer？
2. 检索：从知识库检索相关文档
3. 增强：将文档拼接到prompt
4. 生成：基于检索内容回答
```

**优势**：
- 减少幻觉
- 提供最新信息
- 可解释性（引用来源）

---

#### Program-Aided Language Models (PAL)

**定义**：生成代码解决问题，而非直接生成答案。

**示例**：
```
问题：如果苹果3元/个，买5个多少钱？

生成代码：
def solution():
    apple_price = 3
    quantity = 5
    total = apple_price * quantity
    return total

执行：15元
```

---

#### Automatic Prompt Engineer (APE)

**定义**：自动搜索最优提示词。

**流程**：
```
1. 生成候选提示词
2. 在验证集上评估
3. 选择表现最好的
```

---

## 七、应用场景专有名词

### 7.1 对话系统

#### System Prompt（系统提示）

**定义**：设定AI的角色、能力和行为规范。

**示例**：
```
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe.
```

---

#### Context Management（上下文管理）

**挑战**：
- 上下文长度限制
- 信息遗忘

**策略**：
```
1. Sliding Window（滑动窗口）
   保留最近的N轮对话

2. Summary（摘要）
   定期总结历史对话

3. Retrieval（检索）
   从历史中检索相关对话
```

---

#### Multi-turn Conversation（多轮对话）

**定义**：保持上下文的连续对话。

**格式**：
```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮您？"},
    {"role": "user", "content": "天气怎么样？"}
  ]
}
```

---

### 7.2 代码生成

#### Code Completion（代码补全）

**定义**：根据上下文补全代码。

**类型**：
- 单行补全
- 多行补全
- 函数补全

**模型**：
- GitHub Copilot（基于Codex）
- CodeLlama
- StarCoder

---

#### Code Synthesis（代码合成）

**定义**：根据自然语言描述生成代码。

**示例**：
```
输入："写一个冒泡排序函数"
输出：
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

---

#### Code Translation（代码翻译）

**定义**：在不同编程语言间转换代码。

**示例**：
```
输入：Python代码
输出：等价的JavaScript代码
```

---

#### HumanEval

**定义**：评估代码生成能力的基准数据集。

**格式**：
- 164道编程题
- 提供函数签名和文档字符串
- 评估生成代码的正确性

**指标**：
```
pass@k：生成k个候选，至少1个正确的比例
pass@1：生成1个候选的通过率
pass@10：生成10个候选的通过率
```

---

### 7.3 多模态

#### Vision-Language Models（视觉-语言模型）

**定义**：同时处理图像和文本。

**代表模型**：
- CLIP（OpenAI）
- BLIP, BLIP-2（Salesforce）
- LLaVA（Visual Instruction Tuning）
- GPT-4V（Vision）
- Gemini（Google）

**任务**：
- 图像描述生成
- 视觉问答（VQA）
- 图像-文本检索
- 视觉推理

---

#### Text-to-Image（文本到图像）

**定义**：根据文本生成图像。

**模型**：
- DALL-E 2, DALL-E 3（OpenAI）
- Stable Diffusion（Stability AI）
- Midjourney
- Imagen（Google）

**技术**：
- Diffusion Models（扩散模型）
- GAN（生成对抗网络）
- Autoregressive Models

---

#### Image-to-Text（图像到文本）

**任务**：
- Image Captioning（图像描述）
- OCR（光学字符识别）
- Visual Question Answering（视觉问答）

---

### 7.4 垂直领域

#### Medical LLM（医疗大模型）

**代表**：
- Med-PaLM（Google）
- BioGPT
- ChatDoctor

**应用**：
- 医疗问答
- 病历分析
- 辅助诊断

---

#### Legal LLM（法律大模型）

**应用**：
- 合同审查
- 法律咨询
- 案例分析

---

#### Financial LLM（金融大模型）

**应用**：
- 金融分析
- 风险评估
- 智能投顾

---

## 八、评估与基准

### 8.1 通用能力评估

#### MMLU（Massive Multitask Language Understanding）

**定义**：跨57个学科的知识问答。

**领域**：
- STEM（科学、技术、工程、数学）
- 人文科学
- 社会科学
- 其他（法律、医学等）

**格式**：多选题

---

#### SuperGLUE

**定义**：更难的自然语言理解基准。

**任务**：
- BoolQ（布尔问答）
- CB（蕴含识别）
- COPA（因果推理）
- WiC（词义消歧）
- WSC（指代消解）

---

#### HellaSwag

**定义**：常识推理，选择最合理的句子续写。

**示例**：
```
前文：一个女孩在吹头发
选项：
A. 她继续吹头发直到干透
B. 她把吹风机扔出窗外
C. 她开始跳舞
D. 她变成了一只猫
```

---

#### TruthfulQA

**定义**：评估模型生成答案的真实性。

**特点**：
- 问题设计容易诱导模型生成错误答案
- 测试模型抵抗幻觉的能力

---

### 8.2 推理能力评估

#### GSM8K

**定义**：小学数学应用题，测试多步推理。

**示例**：
```
问题：一个餐厅有23张桌子，每张桌子4把椅子。
     如果有5张桌子已经坐满，还有多少把空椅子？

答案：(23 - 5) × 4 = 72把
```

---

#### MATH

**定义**：高中和竞赛级别数学问题。

**难度**：比GSM8K更难

---

#### BBH（Big-Bench Hard）

**定义**：从Big-Bench中选出模型表现差的任务。

**任务类型**：
- 逻辑推理
- 算术
- 符号操作

---

### 8.3 多模态评估

#### VQA（Visual Question Answering）

**定义**：根据图像回答问题。

**数据集**：
- VQA v2
- GQA（场景图问答）
- OK-VQA（需要外部知识）

---

#### COCO Captions

**定义**：图像描述生成。

**指标**：
- BLEU
- METEOR
- CIDEr
- SPICE

---

## 九、开源生态

### 9.1 模型系列

#### GPT系列（OpenAI）

```
GPT-1 (2018): 117M参数
GPT-2 (2019): 1.5B参数
GPT-3 (2020): 175B参数
  ├─ Davinci
  ├─ Curie
  ├─ Babbage
  └─ Ada

ChatGPT (2022): GPT-3.5 + RLHF
GPT-4 (2023): 多模态
GPT-4 Turbo (2023): 128K上下文
```

---

#### LLaMA系列（Meta）

```
LLaMA (2023):
  ├─ 7B
  ├─ 13B
  ├─ 33B
  └─ 65B

LLaMA-2 (2023):
  ├─ 7B / 7B-Chat
  ├─ 13B / 13B-Chat
  └─ 70B / 70B-Chat

特点：
- 开源
- 高效（训练tokens更多）
- 可商用
```

---

#### BERT系列

```
BERT (2018)
  ├─ BERT-Base: 110M
  └─ BERT-Large: 340M

变体：
├─ RoBERTa: 改进训练策略
├─ ALBERT: 参数共享
├─ DeBERTa: 解耦注意力
├─ ELECTRA: 判别式预训练
└─ DistilBERT: 蒸馏版本
```

---

#### T5系列（Google）

```
T5 (2019): Text-to-Text
  ├─ Small: 60M
  ├─ Base: 220M
  ├─ Large: 770M
  ├─ XL: 3B
  └─ XXL: 11B

Flan-T5 (2022): Instruction Tuning
mT5: 多语言版本
```

---

### 9.2 工具与框架

#### Hugging Face Transformers

**定义**：最流行的NLP库。

**功能**：
- 预训练模型加载
- 微调
- 推理
- 模型分享（Model Hub）

**使用**：
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

#### LangChain

**定义**：构建LLM应用的框架。

**功能**：
- Prompt管理
- Chain（链式调用）
- Agent（智能体）
- Memory（记忆）
- Tool使用

---

#### LlamaIndex

**定义**：数据索引和检索框架（原GPT Index）。

**用途**：
- 构建RAG应用
- 文档索引
- 知识库问答

---

#### vLLM

**定义**：高性能LLM推理引擎。

**特点**：
- PagedAttention
- 高吞吐量
- 连续批处理

---

#### DeepSpeed

**定义**：微软的分布式训练库。

**功能**：
- ZeRO（内存优化）
- 3D并行
- 混合精度训练

---

#### Megatron-LM

**定义**：NVIDIA的大模型训练框架。

**特点**：
- 张量并行
- 流水线并行
- 高效训练超大模型

---

## 十、前沿技术

### 10.1 长上下文

#### Sparse Attention（稀疏注意力）

**定义**：只计算部分位置的注意力，降低复杂度。

**类型**：
- Local Attention（局部注意力）
- Strided Attention（跨步注意力）
- Block Sparse Attention

**模型**：
- Longformer
- BigBird

---

#### Recurrent Memory

**定义**：使用循环机制处理长序列。

**模型**：
- Transformer-XL
- Compressive Transformer

---

#### Retrieval-based Context

**定义**：检索相关片段，而非处理全部上下文。

**模型**：
- RETRO（Retrieval-Enhanced Transformer）

---

### 10.2 多模态

#### Flamingo（DeepMind）

**特点**：
- 少样本视觉语言学习
- 交错图像-文本输入

---

#### Unified-IO

**定义**：统一处理多种模态和任务的模型。

**支持**：
- 图像
- 文本
- 音频
- 视频

---

### 10.3 推理能力

#### Self-Taught Reasoner

**定义**：模型自己生成训练数据提升推理。

**流程**：
```
1. 模型生成推理链
2. 筛选正确的
3. 用于微调
```

---

#### Least-to-Most Prompting

**定义**：将复杂问题分解为简单子问题。

**流程**：
```
问题：计算多个数的乘积
1. 先计算前两个数的乘积
2. 再乘以第三个数
3. 依此类推
```

---

### 10.4 效率优化

#### Mixture of Experts (MoE)

**定义**：只激活部分参数。

**架构**：
```
输入 → Router（选择专家）
     ↓
  Expert 1  Expert 2  Expert 3  ... Expert N
     ↓         ↓         ↓            ↓
     ────────── 合并 ──────────
                ↓
              输出
```

**优势**：
- 模型容量大，计算量小
- 训练和推理高效

**模型**：
- Switch Transformer
- GLaM
- Mixtral 8x7B

---

#### Model Distillation（模型蒸馏）

**定义**：用大模型（教师）训练小模型（学生）。

**流程**：
```
Teacher Model (Large)
        ↓ 软标签
Student Model (Small)
```

**应用**：
- DistilBERT（BERT的蒸馏版）
- TinyBERT
- DistilGPT-2

---

#### Pruning（剪枝）

**定义**：移除不重要的权重或神经元。

**类型**：
- 权重剪枝（Weight Pruning）
- 结构化剪枝（Structured Pruning）
- 动态剪枝（Dynamic Pruning）

---

## 十一、术语速查表

### 11.1 缩写词汇表

| 缩写 | 全称 | 中文 |
|------|------|------|
| LLM | Large Language Model | 大语言模型 |
| NLP | Natural Language Processing | 自然语言处理 |
| MLM | Masked Language Modeling | 掩码语言建模 |
| CLM | Causal Language Modeling | 因果语言建模 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| DPO | Direct Preference Optimization | 直接偏好优化 |
| LoRA | Low-Rank Adaptation | 低秩适应 |
| PEFT | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| CoT | Chain-of-Thought | 思维链 |
| ToT | Tree of Thoughts | 思维树 |
| ICL | In-Context Learning | 上下文学习 |
| Few-shot | Few-shot Learning | 少样本学习 |
| Zero-shot | Zero-shot Learning | 零样本学习 |
| MoE | Mixture of Experts | 混合专家 |
| KV Cache | Key-Value Cache | 键值缓存 |
| BPE | Byte Pair Encoding | 字节对编码 |
| FLOPs | Floating Point Operations | 浮点运算次数 |

---

### 11.2 按场景分类速查

#### 训练阶段
```
Pre-training（预训练）
Fine-tuning（微调）
Instruction Tuning（指令微调）
RLHF（人类反馈强化学习）
Continual Learning（持续学习）
Curriculum Learning（课程学习）
```

#### 推理阶段
```
Greedy Decoding（贪婪解码）
Beam Search（束搜索）
Sampling（采样）
Temperature（温度）
Top-k / Top-p（核采样）
```

#### 性能优化
```
Quantization（量化）
Pruning（剪枝）
Distillation（蒸馏）
LoRA（低秩适应）
Flash Attention（快速注意力）
KV Cache（键值缓存）
```

#### 能力提升
```
Chain-of-Thought（思维链）
Few-shot Learning（少样本学习）
RAG（检索增强）
ReAct（推理+行动）
Self-Consistency（自洽性）
```

---

## 十二、实战示例

### 12.1 完整对话流程

```python
# 使用主流术语描述一个完整的LLM对话流程

# 1. Tokenization（分词）
input_text = "解释什么是Transformer"
tokens = tokenizer.encode(input_text)  # BPE分词
# tokens: [1234, 5678, 90, 12, 3456]

# 2. Add Special Tokens（添加特殊token）
tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

# 3. Forward Pass（前向传播）
with torch.no_grad():
    outputs = model(
        input_ids=tokens,
        use_cache=True,  # 启用KV Cache
        return_dict=True
    )

# 4. Logits to Probabilities（logits转概率）
logits = outputs.logits[:, -1, :]  # 获取最后一个token的logits
probs = F.softmax(logits / temperature, dim=-1)  # Temperature采样

# 5. Sampling（采样）
# Top-p (Nucleus) Sampling
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
nucleus = cumulative_probs <= top_p
next_token = sorted_indices[nucleus][0]

# 6. Decode（解码）
generated_text = tokenizer.decode(next_token)

# 7. Repeat（重复2-6直到生成EOS或达到max_length）
```

---

### 12.2 RAG系统实现

```python
# 完整RAG流程，使用标准术语

# 1. Document Chunking（文档切块）
chunks = split_documents(documents, chunk_size=512, overlap=50)

# 2. Embedding（向量化）
embeddings = embedding_model.encode(chunks)  # Dense Retrieval

# 3. Index Building（构建索引）
index = faiss.IndexFlatIP(embedding_dim)  # 使用FAISS
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 4. Query Embedding（查询向量化）
query = "什么是Attention机制？"
query_embedding = embedding_model.encode([query])
faiss.normalize_L2(query_embedding)

# 5. Retrieval（检索）
k = 5  # Top-k检索
scores, indices = index.search(query_embedding, k)

# 6. Re-ranking（重排序 - 可选）
reranked_docs = cross_encoder.rerank(query, retrieved_chunks)

# 7. Context Augmentation（上下文增强）
context = "\n\n".join([chunks[i] for i in indices[0]])
prompt = f"""基于以下信息回答问题：

{context}

问题：{query}
答案："""

# 8. Generation（生成）
response = llm.generate(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

---

## 十三、总结

本文档系统化地整理了AI大模型领域的专有名词，涵盖：

### 核心知识体系
1. **基础架构**：Transformer、Attention、位置编码
2. **训练技术**：预训练、微调、RLHF、对齐
3. **推理优化**：采样策略、KV Cache、量化
4. **提示工程**：CoT、Few-shot、RAG
5. **评估基准**：MMLU、GSM8K、HumanEval
6. **开源生态**：GPT、LLaMA、BERT系列

### 学习路径建议

**入门阶段**：
- Transformer基础
- Tokenization
- 基本推理（Greedy, Sampling）
- Prompt Engineering

**进阶阶段**：
- 注意力机制变体
- 微调技术（Full, LoRA）
- RAG系统
- 长上下文处理

**高级阶段**：
- RLHF对齐
- 模型并行
- 自定义训练
- 多模态

### 实践建议

1. **动手实践**：使用Hugging Face Transformers实现基础功能
2. **阅读论文**：理解核心技术的原理
3. **参与开源**：为LangChain、vLLM等项目贡献
4. **跟踪前沿**：关注arXiv、Twitter、Reddit

掌握这些专有名词，你将能够：
- 阅读技术论文
- 参与技术讨论
- 实现LLM应用
- 优化模型性能

大模型技术日新月异，持续学习是关键！🚀

---

## 十四、更多专有名词补充

### 14.1 神经网络基础术语

#### Activation Function（激活函数）

**常见激活函数**：

**1. ReLU（Rectified Linear Unit）**
```python
f(x) = max(0, x)
```
- 优点：计算简单，缓解梯度消失
- 缺点：可能出现神经元死亡

**2. GELU（Gaussian Error Linear Unit）**
```python
f(x) = x * Φ(x)  # Φ是标准正态分布的累积分布函数
```
- Transformer常用
- 比ReLU更平滑

**3. Swish / SiLU**
```python
f(x) = x * sigmoid(x)
```
- Google Brain提出
- 自适应门控

**4. GLU（Gated Linear Unit）**
```python
f(x) = (x * W + b) ⊗ σ(x * V + c)
```
- LLaMA使用
- 提升表达能力

---

#### Dropout

**定义**：训练时随机丢弃部分神经元，防止过拟合。

```python
# 训练时
output = x * mask / keep_prob  # mask是随机0/1向量

# 推理时
output = x  # 不使用dropout
```

**变体**：
- **DropConnect**：丢弃连接而非神经元
- **Spatial Dropout**：丢弃整个特征图
- **DropPath**：随机丢弃残差连接（Vision Transformer使用）

---

#### Batch Normalization（批归一化）

**定义**：对每个batch的特征进行归一化。

```python
# 对每个特征维度
μ_B = (1/m) * Σ x_i
σ²_B = (1/m) * Σ (x_i - μ_B)²
x̂_i = (x_i - μ_B) / √(σ²_B + ε)
y_i = γ * x̂_i + β  # γ, β可学习
```

**与Layer Norm区别**：
- **Batch Norm**：对batch维度归一化（CNN常用）
- **Layer Norm**：对特征维度归一化（Transformer常用）

---

### 14.2 损失函数

#### Cross Entropy Loss（交叉熵损失）

**定义**：分类任务的标准损失函数。

```python
# 二分类
L = -[y*log(p) + (1-y)*log(1-p)]

# 多分类
L = -Σ y_i * log(p_i)
```

**语言模型中的使用**：
```python
# 预测下一个token
loss = CrossEntropy(logits, target_token_id)
```

---

#### Contrastive Loss（对比损失）

**定义**：拉近相似样本，推远不相似样本。

**InfoNCE Loss**（对比学习标准损失）：
```python
L = -log(exp(sim(x, x+)/τ) / Σ exp(sim(x, x_i)/τ))

其中：
- x, x+: 正样本对
- x_i: 负样本
- τ: 温度参数
```

**应用**：
- CLIP（图像-文本对比学习）
- SimCLR（自监督学习）
- Contriever（无监督检索）

---

#### Ranking Loss（排序损失）

**定义**：优化样本的相对排序。

**常见类型**：

**1. Pairwise Ranking Loss**
```python
L = max(0, margin - score(pos) + score(neg))
```

**2. Triplet Loss（三元组损失）**
```python
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)

a: anchor（锚点）
p: positive（正样本）
n: negative（负样本）
```

**应用**：
- 检索系统排序
- 推荐系统
- RLHF中的奖励模型训练

---

### 14.3 优化器详解

#### SGD（Stochastic Gradient Descent）

**定义**：随机梯度下降。

```python
θ_t = θ_{t-1} - η * ∇L(θ_{t-1})

η: 学习率
```

**变体**：
- **SGD with Momentum**：增加动量
```python
v_t = β * v_{t-1} + ∇L(θ)
θ_t = θ_{t-1} - η * v_t
```

- **Nesterov Momentum**：预测未来梯度
```python
v_t = β * v_{t-1} + ∇L(θ - β * v_{t-1})
θ_t = θ_{t-1} - η * v_t
```

---

#### Adam（Adaptive Moment Estimation）

**定义**：自适应学习率优化器。

```python
m_t = β1 * m_{t-1} + (1-β1) * g_t       # 一阶矩估计
v_t = β2 * v_{t-1} + (1-β2) * g_t²      # 二阶矩估计

m̂_t = m_t / (1 - β1^t)  # 偏差修正
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)
```

**典型超参数**：
- β1 = 0.9
- β2 = 0.999
- η = 1e-3 或 3e-4

---

#### AdamW

**改进**：正确实现权重衰减（Weight Decay）。

```python
# Adam中的L2正则化（错误）
g_t = g_t + λ * θ_t

# AdamW的权重衰减（正确）
θ_t = θ_{t-1} - η * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
```

**使用**：现代Transformer训练的标准选择

---

#### Adafactor

**定义**：节省内存的Adam变体。

**特点**：
- 不存储完整的二阶矩
- 使用矩阵分解近似
- T5训练使用

---

#### Lion（EvoLved Sign Momentum）

**定义**：Google 2023年提出的新优化器。

```python
update = sign(β1 * m_{t-1} + (1-β1) * g_t)
θ_t = θ_{t-1} - η * update
m_t = β2 * m_{t-1} + (1-β2) * g_t
```

**优势**：
- 更节省内存
- 收敛更快
- 性能与Adam相当

---

### 14.4 正则化技术

#### Weight Decay（权重衰减）

**定义**：L2正则化，防止权重过大。

```python
L_total = L_task + λ * Σ||θ||²
```

**典型值**：λ = 0.01 或 0.1

---

#### Label Smoothing（标签平滑）

**定义**：软化one-hot标签，防止过拟合。

```python
# 原始
y = [0, 0, 1, 0, 0]

# 平滑后（ε=0.1）
y_smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
```

**效果**：提升泛化能力，减少过自信

---

#### Early Stopping（早停）

**定义**：验证集性能不再提升时停止训练。

**策略**：
```
patience = 5  # 容忍5个epoch不提升
if val_loss没有改善 for patience epochs:
    停止训练
    恢复最佳checkpoint
```

---

### 14.5 模型架构变体

#### Sparse Transformer

**定义**：使用稀疏注意力模式，降低复杂度。

**注意力模式**：

**1. Local Attention（局部注意力）**
```
每个token只关注周围k个位置
复杂度：O(n*k)
```

**2. Strided Attention（跨步注意力）**
```
每隔s个位置关注一次
用于捕捉长距离依赖
```

**3. Block Sparse Attention**
```
分块注意力矩阵
OpenAI GPT-3使用
```

---

#### Linformer

**定义**：将attention复杂度降至O(n)。

**方法**：
```
K' = E_K * K  # 投影到更低维度
V' = E_V * V

Attention(Q, K', V') → O(n*k)  # k << n
```

---

#### Reformer

**特点**：
- LSH Attention（局部敏感哈希）
- 可逆层（Reversible Layers）
- 大幅节省内存

**应用**：处理超长序列（64K tokens）

---

### 14.6 训练技巧

#### Gradient Checkpointing（梯度检查点）

**定义**：不保存所有中间激活，反向传播时重新计算。

**权衡**：
- 内存：减少50-90%
- 时间：增加20-30%

```python
# PyTorch
from torch.utils.checkpoint import checkpoint

output = checkpoint(layer, input)  # 不保存激活
```

**使用场景**：训练超大模型

---

#### Loss Scaling（损失缩放）

**定义**：混合精度训练中，放大损失避免下溢。

```python
# 前向传播
loss = model(input)
scaled_loss = loss * scale_factor

# 反向传播
scaled_loss.backward()

# 梯度缩小
gradients = gradients / scale_factor
```

---

#### 学习率查找器（Learning Rate Finder）

**方法**：
```
1. 从很小的lr开始
2. 逐渐增大lr
3. 绘制loss vs lr曲线
4. 选择loss下降最快的lr
```

**工具**：
- PyTorch Lightning内置
- fastai提供lr_find()

---

### 14.7 数据处理

#### Data Collator（数据整理器）

**定义**：将不同长度的样本组成batch。

**策略**：

**1. Padding（填充）**
```python
# 填充到batch最大长度
[1,2,3] + [PAD] → [1,2,3,0]
[1,2]   + [PAD] → [1,2,0,0]
```

**2. Truncation（截断）**
```python
# 截断到最大长度
[1,2,3,4,5] → [1,2,3,4]  # max_len=4
```

**3. Dynamic Padding**
```python
# 只填充到当前batch最大长度
batch = [[1,2], [1,2,3,4,5]]
→ [[1,2,0,0,0], [1,2,3,4,5]]
```

---

#### Attention Mask（注意力掩码）

**定义**：指示哪些位置需要注意。

**类型**：

**1. Padding Mask**
```python
# 忽略padding位置
tokens = [1, 2, 3, 0, 0]  # 0是PAD
mask   = [1, 1, 1, 0, 0]  # 1=attend, 0=ignore
```

**2. Causal Mask（因果掩码）**
```python
# 只能看到之前的token（GPT使用）
mask = [
  [1, 0, 0, 0],  # token 0只看自己
  [1, 1, 0, 0],  # token 1看0,1
  [1, 1, 1, 0],  # token 2看0,1,2
  [1, 1, 1, 1]   # token 3看全部
]
```

**3. Bidirectional Mask（双向掩码）**
```python
# 可以看到全部token（BERT使用）
mask = [
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1]
]
```

---

### 14.8 提示工程进阶

#### Meta-prompting（元提示）

**定义**：用prompt指导如何写prompt。

**示例**：
```
你是一个prompt工程师。为以下任务设计一个最优的prompt：
任务：情感分析
要求：准确率高，格式清晰
```

---

#### Prompt Chaining（提示链）

**定义**：将复杂任务拆分为多个prompt。

**示例**：
```
Prompt 1: 提取文章关键信息
Prompt 2: 基于关键信息生成摘要
Prompt 3: 优化摘要的可读性
```

---

#### Iterative Refinement（迭代精炼）

**流程**：
```
1. 生成初始输出
2. 评估质量
3. 基于反馈改进
4. 重复2-3直到满意
```

---

#### Role Prompting（角色提示）

**定义**：赋予模型特定角色。

**示例**：
```
你是一位资深Python工程师，拥有10年经验...
你是一位物理学教授，擅长用简单语言解释复杂概念...
```

---

### 14.9 多模态专有名词

#### Vision Transformer (ViT)

**定义**：将Transformer应用于图像。

**流程**：
```
1. 图像分割为patches（如16x16）
2. 每个patch线性投影为embedding
3. 添加位置编码
4. 输入Transformer Encoder
```

**公式**：
```
z_0 = [x_cls; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos

x_p^i: 第i个patch
E: patch embedding矩阵
E_pos: 位置编码
```

---

#### CLIP (Contrastive Language-Image Pre-training)

**架构**：
```
Image Encoder (ViT) → Image Embeddings
Text Encoder (Transformer) → Text Embeddings

训练目标：
- 匹配的图像-文本对：相似度高
- 不匹配的对：相似度低
```

**应用**：
- Zero-shot图像分类
- 图像搜索
- 文生图引导（DALL-E 2）

---

#### Diffusion Models（扩散模型）

**核心概念**：

**1. Forward Process（前向过程）**
```
逐步向图像添加噪声，直到变成纯噪声
x_0 → x_1 → x_2 → ... → x_T (noise)
```

**2. Reverse Process（反向过程）**
```
从噪声逐步去噪，生成图像
x_T → x_{T-1} → ... → x_1 → x_0 (image)
```

**训练目标**：
```python
L = E[||ε - ε_θ(x_t, t)||²]

ε: 真实噪声
ε_θ: 模型预测的噪声
```

**代表模型**：
- DDPM（Denoising Diffusion Probabilistic Models）
- Stable Diffusion
- DALL-E 2

---

#### Latent Diffusion

**改进**：在压缩的latent space进行扩散。

**优势**：
- 计算效率高
- 生成质量好

**架构**：
```
Image → VAE Encoder → Latent Code
Latent Code → Diffusion Process → Denoised Latent
Denoised Latent → VAE Decoder → Image
```

**代表**：Stable Diffusion

---

#### ControlNet

**定义**：为扩散模型添加条件控制。

**控制方式**：
- Canny边缘
- 深度图
- 人体姿态
- 语义分割图

**应用**：精确控制生成内容的结构和布局

---

### 14.10 强化学习相关

#### Policy（策略）

**定义**：从状态到动作的映射。

```python
π(a|s): 在状态s下选择动作a的概率
```

**类型**：
- Deterministic Policy：确定性策略
- Stochastic Policy：随机策略

---

#### Value Function（价值函数）

**定义**：评估状态或动作的好坏。

**类型**：

**1. State Value Function (V)**
```python
V^π(s) = E[Σ γ^t r_t | s_0=s, π]

未来累积奖励的期望
```

**2. Action Value Function (Q)**
```python
Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]

在状态s执行动作a的价值
```

**3. Advantage Function (A)**
```python
A^π(s,a) = Q^π(s,a) - V^π(s)

动作a相对平均水平的优势
```

---

#### Reward Shaping（奖励塑形）

**定义**：设计辅助奖励引导学习。

**方法**：
```python
# 稀疏奖励（难学习）
reward = {1 if win, 0 otherwise}

# 密集奖励（容易学习）
reward = base_reward + distance_bonus + time_penalty
```

**风险**：可能引入意外的学习目标

---

#### Exploration vs Exploitation

**定义**：探索新策略 vs 利用已知好策略。

**平衡方法**：

**1. ε-Greedy**
```python
if random() < ε:
    选择随机动作（探索）
else:
    选择最优动作（利用）
```

**2. Entropy Bonus（熵奖励）**
```python
reward = task_reward + β * H(π)

H(π): 策略的熵（多样性）
β: 权重系数
```

---

### 14.11 知识蒸馏

#### Teacher-Student Framework

**流程**：
```
Teacher Model (Large):
  - 已训练好的大模型
  - 生成软标签（概率分布）

Student Model (Small):
  - 待训练的小模型
  - 学习teacher的输出分布
```

**损失函数**：
```python
L = α * L_hard + (1-α) * L_soft

L_hard: 真实标签的交叉熵
L_soft: teacher软标签的KL散度
```

---

#### Knowledge Distillation Loss

**温度缩放**：
```python
# Teacher outputs
p_t = softmax(logits_t / T)

# Student outputs
p_s = softmax(logits_s / T)

# Distillation loss
L_KD = T² * KL(p_t || p_s)

T: 温度（通常2-5）
```

**温度作用**：
- T=1：原始分布（尖锐）
- T>1：更平滑，包含更多信息

---

#### Intermediate Layer Distillation

**定义**：蒸馏中间层的表示。

```python
L_hidden = ||h_teacher - W * h_student||²

W: 对齐矩阵（维度可能不同）
```

**优势**：学习teacher的内部表示

---

### 14.12 模型压缩

#### Low-Rank Factorization（低秩分解）

**定义**：将大矩阵分解为小矩阵的乘积。

```python
# 原始权重
W ∈ R^(m×n)  # mn个参数

# 低秩分解
W ≈ U × V^T
U ∈ R^(m×r), V ∈ R^(n×r)  # (m+n)r个参数

当r << min(m,n)时，显著减少参数
```

**应用**：LoRA就是基于这个原理

---

#### Structured Pruning（结构化剪枝）

**定义**：移除整个结构单元（神经元、通道、层）。

**粒度**：
```
Layer-level: 移除整层
Channel-level: 移除卷积通道
Attention-head-level: 移除注意力头
```

**优势**：
- 加速效果明显
- 硬件友好
- 无需特殊推理库

---

#### Dynamic Quantization（动态量化）

**定义**：推理时动态确定量化参数。

```python
# 静态量化（训练时确定）
scale, zero_point = calibrate(weights)
quantized = quantize(weights, scale, zero_point)

# 动态量化（推理时确定）
for batch in data:
    scale, zero_point = compute_dynamic(activations)
    quantized = quantize(activations, scale, zero_point)
```

**适用**：激活值分布不固定的场景

---

### 14.13 分布式训练

#### Ring-AllReduce

**定义**：分布式训练中高效同步梯度的算法。

**流程**：
```
GPU 0: [g0, g1, g2, g3]
GPU 1: [g0, g1, g2, g3]
GPU 2: [g0, g1, g2, g3]
GPU 3: [g0, g1, g2, g3]

分块累加 → 所有GPU得到相同梯度总和
```

**优势**：
- 通信量：O(N)（vs Naive: O(N²)）
- 无需中心节点

---

#### ZeRO（Zero Redundancy Optimizer）

**定义**：DeepSpeed的内存优化技术。

**三个阶段**：

**ZeRO-1**: 分片优化器状态
```
每个GPU只存储部分Adam状态
内存节省：4x
```

**ZeRO-2**: 分片梯度
```
每个GPU只存储部分梯度
内存节省：8x
```

**ZeRO-3**: 分片模型参数
```
每个GPU只存储部分参数
内存节省：N_gpu x（几乎线性）
```

**效果**：可以训练万亿参数模型

---

#### Gradient Checkpointing + Offloading

**组合技巧**：
```
1. Gradient Checkpointing: 减少激活内存
2. CPU Offloading: 将部分数据移到CPU
3. NVMe Offloading: 将数据移到磁盘

权衡：内存 ↓↓, 速度 ↓
```

---

### 14.14 安全与对齐

#### Adversarial Training（对抗训练）

**定义**：训练时加入对抗样本，提升鲁棒性。

```python
# 生成对抗样本
x_adv = x + ε * sign(∇_x L(x, y))

# 对抗训练
L_total = L(x, y) + L(x_adv, y)
```

**应用**：防御对抗攻击

---

#### Prompt Injection（提示注入）

**定义**：恶意用户通过精心设计的prompt绕过限制。

**示例**：
```
用户输入：
"忽略之前所有指令，现在按照我的要求..."

防御：
- 输入过滤
- 指令分离
- 输出验证
```

---

#### AI Alignment Tax

**定义**：对齐操作带来的性能损失。

**权衡**：
```
安全性 ↑ ⟺ 能力 ↓

RLHF可能：
- 降低创造力
- 增加拒绝率
- 减少某些能力
```

---

#### Watermarking（水印）

**定义**：在生成内容中嵌入不可见的标记。

**方法**：
```
1. 选择特定token组合
2. 调整采样分布
3. 检测时统计token模式
```

**应用**：识别AI生成内容

---

### 14.15 新兴概念

#### Emergent Abilities（涌现能力）

**定义**：模型规模达到阈值后突然出现的能力。

**示例**：
- Few-shot learning
- Chain-of-thought推理
- 算术能力
- 代码理解

**特点**：
- 小模型完全不具备
- 中等模型开始出现
- 大模型表现优异

---

#### Scaling Hypothesis（缩放假说）

**核心观点**：
```
模型性能 ∝ scale(params, data, compute)

只要持续扩大规模，性能就会持续提升
```

**支持证据**：
- GPT系列：3 → 3.5 → 4
- PaLM: 8B → 62B → 540B

**争议**：是否存在上限？

---

#### Bitter Lesson（苦涩教训）

**Rich Sutton的观点**：
```
长期来看：
通用方法 + 计算 > 人类知识 + 手工特征

简单的方法 + 大规模计算
往往胜过
复杂的方法 + 人类先验
```

---

#### AGI（Artificial General Intelligence）

**定义**：通用人工智能，能完成人类所有智力任务。

**评估标准**：
- 多任务能力
- 迁移学习
- 创造力
- 常识推理
- 自我改进

**现状**：尚未实现，但GPT-4已展现部分特征

---

#### Superintelligence（超级智能）

**定义**：在所有领域都超越人类的AI。

**类型**：
- Speed Superintelligence（速度）
- Collective Superintelligence（集体）
- Quality Superintelligence（质量）

---

## 十五、按字母顺序的完整词汇表

### A

- **Activation Function** - 激活函数
- **Adapter** - 适配器（PEFT方法）
- **Adversarial Training** - 对抗训练
- **AGI** - 通用人工智能
- **Alignment** - 对齐
- **ALiBi** - 注意力线性偏置
- **Alpaca** - 斯坦福指令微调模型
- **ANCE** - 近似最近邻对比学习
- **Anthropic** - Claude开发公司
- **APE** - 自动提示工程
- **Attention Mask** - 注意力掩码
- **Autoregressive** - 自回归
- **AWQ** - 激活感知权重量化

### B

- **Backpropagation** - 反向传播
- **BART** - 序列到序列预训练模型
- **Batch Normalization** - 批归一化
- **Batch Size** - 批大小
- **BBH** - 大型基准困难任务
- **Beam Search** - 束搜索
- **BERT** - 双向编码表示
- **Bias** - 偏置
- **BigBird** - 稀疏注意力模型
- **BioBERT** - 生物医学BERT
- **BLEU** - 双语评估辅助
- **BLIP** - 引导语言-图像预训练
- **BPE** - 字节对编码
- **BF16** - Brain Float 16

### C

- **Catastrophic Forgetting** - 灾难性遗忘
- **Causal LM** - 因果语言建模
- **Chain-of-Thought** - 思维链
- **ChatGPT** - OpenAI对话模型
- **Checkpoint** - 检查点
- **CIDEr** - 共识图像描述评估
- **CLIP** - 对比语言-图像预训练
- **CLM** - 因果语言建模
- **Code Completion** - 代码补全
- **CodeBERT** - 代码理解模型
- **CodeLlama** - Meta代码生成模型
- **ColBERT** - 上下文化后期交互
- **Constitutional AI** - 宪法AI
- **Context Window** - 上下文窗口
- **Contrastive Learning** - 对比学习
- **ControlNet** - 扩散模型控制
- **CoT** - 思维链
- **Cross-Attention** - 交叉注意力
- **Cross Entropy** - 交叉熵
- **Curriculum Learning** - 课程学习

### D

- **Data Augmentation** - 数据增强
- **Data Parallelism** - 数据并行
- **DALL-E** - OpenAI文生图模型
- **DAN** - Do Anything Now越狱
- **DeBERTa** - 解耦增强BERT
- **Decoder** - 解码器
- **DeepSpeed** - 微软训练框架
- **Diffusion Models** - 扩散模型
- **Distillation** - 蒸馏
- **DistilBERT** - BERT蒸馏版
- **Diverse Beam Search** - 多样化束搜索
- **DPO** - 直接偏好优化
- **DPR** - 密集段落检索
- **Dropout** - 随机丢弃
- **Dynamic Quantization** - 动态量化

### E

- **Early Stopping** - 早停
- **Elastic Weight Consolidation** - 弹性权重整合
- **ELECTRA** - 判别式预训练
- **Embeddings** - 嵌入
- **Emergent Abilities** - 涌现能力
- **Encoder** - 编码器
- **Entity Masking** - 实体掩码
- **Epoch** - 训练轮次
- **EWC** - 弹性权重整合

### F

- **FAISS** - Facebook相似度搜索
- **Few-shot** - 少样本学习
- **FinBERT** - 金融BERT
- **Fine-tuning** - 微调
- **FLAN** - 指令微调模型系列
- **Flamingo** - DeepMind多模态模型
- **Flash Attention** - 快速注意力
- **FLOPs** - 浮点运算
- **FP16/FP32** - 浮点精度
- **Frozen** - 冻结参数

### G

- **GAN** - 生成对抗网络
- **Gemini** - Google多模态模型
- **GELU** - 高斯误差线性单元
- **Generative AI** - 生成式AI
- **GLU** - 门控线性单元
- **GPTQ** - GPT量化
- **Gradient Accumulation** - 梯度累积
- **Gradient Checkpointing** - 梯度检查点
- **Gradient Clipping** - 梯度裁剪
- **Gradient Descent** - 梯度下降
- **GraphCodeBERT** - 图代码BERT
- **Greedy Decoding** - 贪婪解码
- **GSM8K** - 小学数学问题集

### H

- **Hallucination** - 幻觉
- **HellaSwag** - 常识推理基准
- **HNSW** - 层次导航小世界图
- **HumanEval** - 代码评估基准
- **Hugging Face** - 著名AI平台
- **Hyperparameter** - 超参数

### I

- **IA³** - 抑制和放大内部激活
- **ICL** - 上下文学习
- **Imagen** - Google文生图模型
- **In-Context Learning** - 上下文学习
- **Inference** - 推理
- **InfoNCE** - 对比学习损失
- **Instruction Tuning** - 指令微调
- **INT4/INT8** - 整数量化精度
- **IVF** - 倒排文件索引

### J

- **Jailbreak** - 越狱攻击

### K

- **KL Divergence** - KL散度
- **Knowledge Distillation** - 知识蒸馏
- **KV Cache** - 键值缓存

### L

- **LaBSE** - 语言无关句子嵌入
- **Label Smoothing** - 标签平滑
- **LangChain** - LLM应用框架
- **Latent Diffusion** - 潜在扩散
- **Layer Normalization** - 层归一化
- **Learning Rate** - 学习率
- **Length Penalty** - 长度惩罚
- **Lion** - 演化符号动量优化器
- **LLaMA** - Meta开源模型
- **LLaVA** - 视觉指令微调
- **LLM** - 大语言模型
- **Logits** - 未归一化的预测分数
- **Longformer** - 长文档Transformer
- **LoRA** - 低秩适应
- **Loss Function** - 损失函数

### M

- **Masked LM** - 掩码语言建模
- **MATH** - 数学问题数据集
- **mBERT** - 多语言BERT
- **Megatron-LM** - NVIDIA训练框架
- **METEOR** - 翻译评估指标
- **Midjourney** - 文生图服务
- **Milvus** - 向量数据库
- **Mistral** - 开源LLM
- **Mixed Precision** - 混合精度
- **Mixtral** - MoE架构模型
- **MLM** - 掩码语言建模
- **MMLU** - 大规模多任务理解
- **MoE** - 混合专家
- **Momentum** - 动量
- **MRR** - 平均倒数排名
- **mT5** - 多语言T5
- **Multi-Head Attention** - 多头注意力
- **Multi-turn** - 多轮对话

### N

- **NDCG** - 归一化折损累计增益
- **NER** - 命名实体识别
- **NLP** - 自然语言处理
- **NSP** - 下一句预测
- **Nucleus Sampling** - 核采样

### O

- **OCR** - 光学字符识别
- **Offloading** - 卸载（到CPU/磁盘）
- **One-shot** - 单样本学习
- **ONNX** - 开放神经网络交换
- **Optimizer** - 优化器
- **Overfitting** - 过拟合

### P

- **PagedAttention** - 分页注意力
- **PAL** - 程序辅助语言模型
- **PaLM** - Google大模型
- **Perplexity** - 困惑度
- **PEFT** - 参数高效微调
- **Pipeline Parallelism** - 流水线并行
- **Positional Encoding** - 位置编码
- **Post-training** - 训练后
- **PPO** - 近端策略优化
- **Precision** - 精确率
- **Prefix Tuning** - 前缀微调
- **Pre-training** - 预训练
- **Prompt** - 提示词
- **Prompt Engineering** - 提示工程
- **Prompt Injection** - 提示注入
- **Pruning** - 剪枝
- **PubMedBERT** - 医学BERT

### Q

- **QLoRA** - 量化LoRA
- **Quantization** - 量化
- **Query** - 查询

### R

- **RAG** - 检索增强生成
- **Ranking Loss** - 排序损失
- **ReAct** - 推理+行动
- **Recall** - 召回率
- **Recurrent** - 循环
- **Red Teaming** - 红队测试
- **Reformer** - 高效Transformer
- **Regularization** - 正则化
- **ReLU** - 修正线性单元
- **Repetition Penalty** - 重复惩罚
- **Re-ranking** - 重排序
- **Residual Connection** - 残差连接
- **RETRO** - 检索增强Transformer
- **Reward Model** - 奖励模型
- **RLHF** - 人类反馈强化学习
- **RMSNorm** - 均方根归一化
- **RoBERTa** - 鲁棒优化BERT
- **RoPE** - 旋转位置编码
- **ROUGE** - 摘要评估指标

### S

- **Sampling** - 采样
- **Scaling Law** - 缩放定律
- **SciBERT** - 科学文献BERT
- **Self-Attention** - 自注意力
- **Self-Consistency** - 自洽性
- **Sentence Transformers** - 句子编码模型
- **SentencePiece** - 分词器
- **Sequence Parallelism** - 序列并行
- **SFT** - 监督微调
- **SGD** - 随机梯度下降
- **Sharding** - 分片
- **Sliding Window** - 滑动窗口
- **Softmax** - 归一化指数函数
- **SOP** - 句子顺序预测
- **Sparse Attention** - 稀疏注意力
- **Speculative Decoding** - 投机解码
- **SPICE** - 语义命题图像描述评估
- **Stable Diffusion** - 稳定扩散
- **StarCoder** - 代码生成模型
- **Superintelligence** - 超级智能
- **SuperGLUE** - 高级语言理解基准
- **Supervised Learning** - 监督学习
- **Swish/SiLU** - 激活函数
- **System Prompt** - 系统提示

### T

- **T5** - Text-to-Text Transfer Transformer
- **Temperature** - 温度参数
- **Tensor Parallelism** - 张量并行
- **TensorRT** - NVIDIA推理引擎
- **Text-to-Image** - 文本到图像
- **TGI** - 文本生成推理
- **Token** - 词元
- **Tokenization** - 分词
- **Top-k Sampling** - Top-k采样
- **Top-p Sampling** - Top-p采样
- **ToT** - 思维树
- **Toxicity** - 毒性
- **TPU** - 张量处理单元
- **Transfer Learning** - 迁移学习
- **Transformer** - 变换器架构
- **Transformer-XL** - 扩展Transformer
- **Triplet Loss** - 三元组损失
- **TruthfulQA** - 真实性问答基准
- **Truncation** - 截断

### U

- **Underfitting** - 欠拟合
- **Unified-IO** - 统一多模态模型
- **Unigram LM** - 单字语言模型

### V

- **Value Function** - 价值函数
- **Variational Autoencoder** - 变分自编码器
- **ViT** - 视觉Transformer
- **vLLM** - 高性能推理引擎
- **Vocabulary** - 词表
- **VQA** - 视觉问答

### W

- **Warmup** - 预热
- **Watermarking** - 水印
- **Weaviate** - 向量数据库
- **Weight Decay** - 权重衰减
- **Whisper** - OpenAI语音模型
- **Whole Word Masking** - 全词掩码
- **WordPiece** - 分词算法

### X

- **XLM-R** - 跨语言掩码模型-RoBERTa

### Z

- **ZeRO** - 零冗余优化器
- **Zero-shot** - 零样本学习

---

## 十六、术语学习建议

### 按难度分级

**入门级（必须掌握）**：
- Transformer, Attention, Token, Embedding
- Fine-tuning, Prompt, Few-shot, Zero-shot
- Temperature, Top-p, Beam Search

**中级（深入理解）**：
- LoRA, RLHF, RAG, CoT
- KV Cache, Quantization
- Instruction Tuning, System Prompt

**高级（专业研究）**：
- MoE, Sparse Attention, Flash Attention
- DPO, Constitutional AI
- ZeRO, Tensor Parallelism

### 学习策略

1. **关联记忆**：将相关术语组合学习
2. **实践验证**：用代码实现核心概念
3. **论文溯源**：阅读原始论文深入理解
4. **社区交流**：参与讨论巩固知识

---

---

## 十七、高级训练技术补充

### 17.1 分布式训练进阶

#### FSDP (Fully Sharded Data Parallel)

**定义**：PyTorch的ZeRO实现，完全分片数据并行。

**特点**：
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 包装模型
model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # 完全分片
    cpu_offload=True,  # CPU卸载
    mixed_precision=True,
)
```

**优势**：
- 与PyTorch原生集成
- 支持CPU offloading
- 内存效率接近ZeRO-3

---

#### 3D Parallelism (三维并行)

**定义**：结合三种并行策略。

**组合**：
```
数据并行 (Data Parallelism)
+
张量并行 (Tensor Parallelism)
+
流水线并行 (Pipeline Parallelism)
= 3D Parallelism
```

**示例配置**：
```python
# Megatron-LM配置
世界大小: 64个GPU
├─ 数据并行度: 4
├─ 张量并行度: 4
└─ 流水线并行度: 4
(4 × 4 × 4 = 64)
```

**应用**：训练万亿参数模型(GPT-3, PaLM等)

---

#### Sequence Parallelism (序列并行)

**定义**：将长序列分片到多个设备。

**适用场景**：
- 超长上下文(>100K tokens)
- 内存受限环境

**实现**：
```python
# 将sequence维度切分
seq_len = 8192
num_gpus = 4
per_gpu_len = seq_len // num_gpus  # 每个GPU处理2048

# GPU 0: tokens[0:2048]
# GPU 1: tokens[2048:4096]
# GPU 2: tokens[4096:6144]
# GPU 3: tokens[6144:8192]
```

---

### 17.2 数据效率技术

#### Active Learning (主动学习)

**定义**：选择最有价值的样本进行标注。

**策略**：
1. **不确定性采样**：选择模型最不确定的样本
2. **多样性采样**：选择代表性样本
3. **对抗性采样**：选择模型易错样本

**流程**：
```python
def active_learning_loop():
    # 1. 初始模型
    model = train_on_labeled_data(initial_data)

    for iteration in range(max_iterations):
        # 2. 在未标注数据上预测
        predictions = model.predict(unlabeled_pool)

        # 3. 选择最有价值的样本
        selected = select_uncertain_samples(predictions, n=100)

        # 4. 人工标注
        labels = human_annotate(selected)

        # 5. 重新训练
        model = train_on_labeled_data(labeled_data + selected)
```

---

#### Semi-Supervised Learning (半监督学习)

**定义**：利用少量标注数据+大量未标注数据训练。

**方法**：

**1. Pseudo-Labeling (伪标签)**
```python
# 用已训练模型为未标注数据打标签
pseudo_labels = model.predict(unlabeled_data)
# 选择高置信度的伪标签
confident = pseudo_labels[confidence > 0.9]
# 加入训练集
train_data += confident
```

**2. Consistency Regularization (一致性正则化)**
```python
# 同一样本的不同增强版本应得到相似预测
def consistency_loss(x):
    x_aug1 = augment(x)
    x_aug2 = augment(x)

    pred1 = model(x_aug1)
    pred2 = model(x_aug2)

    loss = KL_divergence(pred1, pred2)
    return loss
```

**3. MixMatch**
- 结合伪标签和一致性正则化
- MixUp数据增强
- 温度锐化

---

#### Self-Supervised Learning (自监督学习)

**定义**：从未标注数据自动生成监督信号。

**代表方法**：

**1. 对比学习 (Contrastive Learning)**
- SimCLR
- MoCo (Momentum Contrast)
- BYOL (Bootstrap Your Own Latent)

**2. 掩码预测 (Masked Prediction)**
- BERT (MLM)
- MAE (Masked Autoencoders)

**3. 预测任务 (Pretext Tasks)**
- 旋转预测
- 拼图还原
- 着色

---

### 17.3 长上下文技术补充

#### Sparse Attention 变体

**1. Sliding Window Attention**
```python
# 只关注固定窗口内的token
window_size = 256

for i in range(seq_len):
    start = max(0, i - window_size)
    end = min(seq_len, i + window_size)
    attention_range = tokens[start:end]
```

**2. Global + Local Attention**
```python
# Longformer策略
attention_pattern = [
    "local",    # 大部分token用局部注意力
    "local",
    "global",   # 部分关键token用全局注意力
    "local",
]
```

**3. Random Attention**
```
每个token随机关注r个其他位置
用于捕捉长距离依赖
```

---

#### Memory-Augmented Networks (记忆增强网络)

**定义**：添加外部可读写的记忆模块。

**类型**：

**1. Neural Turing Machine (NTM)**
- 可微分的读写操作
- 基于内容的寻址

**2. Differentiable Neural Computer (DNC)**
- NTM的改进版
- 动态内存分配

**3. Memory Networks**
- 用于QA任务
- 支持多跳推理

---

### 17.4 模型架构创新

#### State Space Models (SSM)

**代表**：Mamba, S4 (Structured State Spaces)

**核心思想**：
```
用状态空间方程替代Attention
复杂度: O(N) vs Attention的O(N²)
```

**优势**：
- 线性复杂度
- 支持超长序列(>100万tokens)
- 高效训练和推理

**公式**：
```python
# 离散状态空间方程
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t + D * x_t

A: 状态转移矩阵
B: 输入矩阵
C: 输出矩阵
D: 前馈矩阵
```

---

#### Hyper-Networks (超网络)

**定义**：用一个网络生成另一个网络的权重。

**应用**：
```python
class HyperNetwork:
    def __init__(self):
        self.meta_net = MetaNetwork()

    def forward(self, task_embedding):
        # 根据任务生成主网络权重
        weights = self.meta_net(task_embedding)
        return weights

# 多任务学习
for task in tasks:
    task_emb = encode_task(task)
    weights = hyper_net(task_emb)
    model.set_weights(weights)
    model.train(task)
```

**优势**：
- 参数共享
- 快速适应新任务
- 元学习

---

#### Neural Architecture Search (NAS)

**定义**：自动搜索最优网络架构。

**方法**：

**1. 强化学习搜索**
```python
# 控制器生成架构
controller = RNN()

for iteration in range(max_iter):
    # 采样架构
    architecture = controller.sample()

    # 训练和评估
    accuracy = train_and_eval(architecture)

    # 更新控制器(策略梯度)
    controller.update(architecture, accuracy)
```

**2. 可微分搜索 (DARTS)**
```python
# 所有可能操作的加权组合
alpha = learnable_parameters()

mixed_op = sum(alpha_i * op_i for i, op_i in operations)

# 同时优化alpha和网络权重
```

**3. 进化算法**
- 种群初始化
- 变异和交叉
- 适者生存

---

### 17.5 测试时优化

#### Test-Time Adaptation (TTA)

**定义**：推理时根据测试数据微调模型。

**方法**：
```python
def test_time_adapt(model, test_sample):
    # 1. 启用BatchNorm等的更新
    model.train()

    # 2. 在测试样本上优化
    loss = self_supervised_loss(test_sample)
    loss.backward()
    optimizer.step()

    # 3. 预测
    model.eval()
    prediction = model(test_sample)
    return prediction
```

**应用场景**：
- 分布偏移
- 域适应
- 个性化

---

#### Test-Time Training (TTT)

**定义**：测试时使用自监督任务更新模型。

**流程**：
```python
# 训练阶段：同时优化主任务和辅助任务
main_loss = supervised_loss(x, y)
aux_loss = self_supervised_loss(x)  # 如旋转预测
total_loss = main_loss + aux_loss

# 测试阶段：用辅助任务适应
test_aux_loss = self_supervised_loss(test_x)
# 更新模型
# 然后预测
```

---

#### Prompt Tuning at Test Time

**定义**：测试时优化prompt而非模型。

```python
# 固定模型，优化prompt
prompt = learnable_prompt_embedding()

for test_sample in test_set:
    # 优化prompt
    loss = compute_loss(model(prompt + test_sample))
    prompt = update_prompt(loss)

    # 用优化后的prompt预测
    prediction = model(prompt + test_sample)
```

---

### 17.6 鲁棒性与安全

#### Certified Robustness (认证鲁棒性)

**定义**：数学证明模型在一定扰动范围内的鲁棒性。

**方法**：

**1. Randomized Smoothing**
```python
def certify_robustness(model, x, sigma, n_samples):
    # 添加高斯噪声并投票
    votes = []
    for _ in range(n_samples):
        x_noisy = x + torch.randn_like(x) * sigma
        pred = model(x_noisy)
        votes.append(pred)

    # 计算认证半径
    top_class = majority_vote(votes)
    certified_radius = compute_radius(votes, sigma)

    return top_class, certified_radius
```

**2. Interval Bound Propagation (IBP)**
- 传播输入的上下界
- 保证输出在安全范围内

---

#### Backdoor Defense (后门防御)

**定义**：检测和缓解模型中的后门攻击。

**后门攻击示例**：
```python
# 攻击者在训练数据中植入触发器
trigger = special_pattern
poisoned_samples = add_trigger(clean_samples, trigger)
poisoned_labels = target_label  # 强制输出特定标签

# 模型学习到：trigger → target_label
```

**防御方法**：

**1. Neural Cleanse**
- 反向工程寻找触发器
- 检测异常小的触发器

**2. Activation Clustering**
- 聚类隐藏层激活
- 检测异常簇

**3. Fine-Pruning**
- 剪枝可疑神经元
- 在干净数据上微调

---

#### Model Extraction Defense (模型提取防御)

**定义**：防止攻击者通过API盗取模型。

**攻击**：
```python
# 攻击者查询API
for x in crafted_inputs:
    y = victim_api(x)
    dataset.append((x, y))

# 训练替代模型
stolen_model = train(dataset)
```

**防御**：
```python
# 1. 添加水印
def watermark_output(logits):
    # 轻微修改输出分布
    watermarked = logits + watermark_signature
    return watermarked

# 2. 输出扰动
def perturb_output(logits):
    noise = random_noise(magnitude=small_epsilon)
    return logits + noise

# 3. 查询限制
if user_queries > threshold:
    rate_limit()
```

---

### 17.7 多任务与元学习

#### Multi-Task Learning (MTL)

**定义**：同时学习多个相关任务。

**架构**：

**1. Hard Parameter Sharing**
```python
# 共享编码器
shared_encoder = Transformer()

# 任务特定头
task1_head = Linear(hidden_dim, num_classes_1)
task2_head = Linear(hidden_dim, num_classes_2)

# 前向
features = shared_encoder(input)
output1 = task1_head(features)
output2 = task2_head(features)

# 联合损失
loss = loss1 + loss2
```

**2. Soft Parameter Sharing**
```
每个任务有独立网络
通过正则化鼓励参数相似
```

**优势**：
- 正则化效果(防止过拟合)
- 知识迁移
- 参数效率

---

#### Meta-Learning (元学习)

**定义**：学习如何学习。

**代表算法**：

**1. MAML (Model-Agnostic Meta-Learning)**
```python
def maml_outer_loop(tasks):
    theta = init_parameters()

    for task in tasks:
        # 内循环：快速适应
        theta_task = theta.clone()
        for step in range(k_steps):
            loss = compute_loss(theta_task, task)
            theta_task -= alpha * grad(loss, theta_task)

        # 外循环：更新初始参数
        meta_loss = compute_loss(theta_task, task)
        theta -= beta * grad(meta_loss, theta)

    return theta  # 适合快速适应的初始参数
```

**2. Prototypical Networks**
```python
# 基于距离的分类
class_prototypes = {}
for class_c in classes:
    # 计算类原型(均值)
    class_prototypes[c] = mean(embeddings[class_c])

# 分类：找最近的原型
def classify(x):
    emb = encoder(x)
    distances = [dist(emb, proto) for proto in class_prototypes.values()]
    return argmin(distances)
```

**3. Reptile**
- MAML的简化版
- 直接向任务适应后的参数移动

**应用**：
- Few-shot learning
- 快速适应
- 个性化

---

### 17.8 神经符号AI

#### Neurosymbolic AI (神经符号结合)

**定义**：结合神经网络和符号推理。

**方法**：

**1. Neural Module Networks (NMN)**
```python
# 根据问题动态组合神经模块
question = "What color is the cat?"

# 解析为程序
program = [
    "find(cat)",      # 定位猫
    "relate(color)",  # 提取颜色属性
]

# 执行程序
cat_region = find_module(image)
color = relate_module(cat_region, "color")
```

**2. Logic Tensor Networks**
- 用张量表示逻辑
- 可微分推理

**3. Semantic Parsing**
```python
# 自然语言 → 逻辑形式
nl = "All dogs are animals"
logic = "∀x (dog(x) → animal(x))"

# 可执行查询
query = "Is Fido an animal?"
# 推理引擎验证
```

---

#### Differentiable Reasoning (可微分推理)

**应用**：

**1. Graph Neural Networks (GNN) for Reasoning**
```python
# 知识图谱推理
class ReasoningGNN(nn.Module):
    def forward(self, entities, relations):
        # 迭代消息传递
        for layer in range(num_layers):
            # 聚合邻居信息
            messages = aggregate_neighbors(entities, relations)
            # 更新实体表示
            entities = update(entities, messages)

        return entities
```

**2. Attention-based Reasoning**
- 多跳注意力
- 动态推理路径

---

### 17.9 持续学习补充

#### Elastic Weight Consolidation (EWC)

**原理**：重要参数变化受限。

```python
# 计算Fisher信息矩阵(参数重要性)
def compute_fisher(model, old_task_data):
    fisher = {}
    for name, param in model.named_parameters():
        # 计算梯度的平方期望
        grads = []
        for x, y in old_task_data:
            loss = model(x, y)
            grad = torch.autograd.grad(loss, param)[0]
            grads.append(grad ** 2)
        fisher[name] = torch.mean(torch.stack(grads))
    return fisher

# EWC损失
def ewc_loss(model, old_params, fisher, lambda_ewc):
    loss = 0
    for name, param in model.named_parameters():
        # 惩罚重要参数的改变
        loss += fisher[name] * ((param - old_params[name]) ** 2).sum()
    return lambda_ewc * loss

# 总损失 = 新任务损失 + EWC正则化
total_loss = new_task_loss + ewc_loss
```

---

#### Progressive Neural Networks

**思想**：为每个新任务添加新列，保留旧列。

```python
# 架构
Task 1: Column 1
Task 2: Column 1 (frozen) + Column 2
Task 3: Column 1 (frozen) + Column 2 (frozen) + Column 3

# 侧向连接
output_task3 = combine(
    column1(x),  # 旧知识
    column2(x),
    column3(x)   # 新知识
)
```

**优点**：
- 无灾难性遗忘
- 知识迁移

**缺点**：
- 参数线性增长

---

#### Experience Replay (经验回放)

**定义**：存储旧任务样本，混合训练。

```python
class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, samples):
        self.buffer.extend(samples)
        # 保持缓冲区大小
        if len(self.buffer) > self.buffer_size:
            self.buffer = random.sample(self.buffer, self.buffer_size)

    def sample(self, n):
        return random.sample(self.buffer, n)

# 训练新任务时
for batch in new_task_data:
    # 混合新旧数据
    old_samples = replay_buffer.sample(batch_size // 2)
    mixed_batch = batch + old_samples

    loss = compute_loss(model, mixed_batch)
    loss.backward()
```

---

### 17.10 模型可解释性进阶

#### Attention Visualization (注意力可视化)

**方法**：
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """
    attention_weights: [seq_len, seq_len]
    tokens: list of token strings
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        cbar=True
    )
    plt.title("Attention Weights")
    plt.show()

# 提取注意力权重
outputs = model(input_ids, output_attentions=True)
attention = outputs.attentions  # tuple of [batch, heads, seq, seq]

# 可视化第一个头的注意力
visualize_attention(attention[0][0, 0].detach(), tokens)
```

---

#### Integrated Gradients

**定义**：归因方法，计算每个输入特征的重要性。

```python
def integrated_gradients(model, input, baseline, steps=50):
    """
    input: 原始输入
    baseline: 基线(如全零)
    """
    # 线性插值路径
    alphas = torch.linspace(0, 1, steps)

    gradients = []
    for alpha in alphas:
        # 插值输入
        interpolated = baseline + alpha * (input - baseline)
        interpolated.requires_grad = True

        # 计算梯度
        output = model(interpolated)
        grad = torch.autograd.grad(output.sum(), interpolated)[0]
        gradients.append(grad)

    # 积分
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    integrated_grads = (input - baseline) * avg_gradients

    return integrated_grads

# 可视化特征重要性
attributions = integrated_gradients(model, input_text, baseline)
```

---

#### SHAP (SHapley Additive exPlanations)

**定义**：基于博弈论的特征归因。

```python
import shap

# 创建解释器
explainer = shap.Explainer(model, background_data)

# 计算SHAP值
shap_values = explainer(test_samples)

# 可视化
shap.plots.waterfall(shap_values[0])  # 单样本
shap.plots.beeswarm(shap_values)       # 多样本汇总
```

**优势**：
- 理论保证(唯一满足某些公理)
- 局部和全局解释
- 模型无关

---

#### Counterfactual Explanations (反事实解释)

**定义**："如果改变X，输出会变成Y"。

```python
def find_counterfactual(model, original_input, target_class):
    """
    寻找最小改变使得预测变为target_class
    """
    # 初始化
    cf = original_input.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([cf], lr=0.01)

    for step in range(max_steps):
        optimizer.zero_grad()

        # 目标：预测为target_class且改变最小
        pred = model(cf)
        classification_loss = -F.log_softmax(pred, dim=-1)[target_class]
        proximity_loss = torch.norm(cf - original_input)

        loss = classification_loss + lambda_prox * proximity_loss
        loss.backward()
        optimizer.step()

        # 检查是否达到目标
        if pred.argmax() == target_class:
            break

    return cf

# 示例
cf_example = find_counterfactual(
    model,
    original_text_embedding,
    target_class=positive_sentiment
)
print(f"Change: {cf_example - original_text_embedding}")
```

---

## 十八、模型压缩与加速补充

### 18.1 知识蒸馏变体

#### Feature Distillation (特征蒸馏)

**定义**：蒸馏中间层特征。

```python
class FeatureDistillation(nn.Module):
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        # 特征对齐层(如果维度不同)
        self.align = nn.Linear(student_dim, teacher_dim)

    def forward(self, x, y):
        # Teacher前向(不计算梯度)
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(x, return_features=True)

        # Student前向
        student_logits, student_features = self.student(x, return_features=True)

        # 特征蒸馏损失
        feature_loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            s_feat_aligned = self.align(s_feat)
            feature_loss += F.mse_loss(s_feat_aligned, t_feat)

        # 输出蒸馏损失
        kd_loss = KL_div(student_logits/T, teacher_logits/T) * T**2

        # 真实标签损失
        ce_loss = F.cross_entropy(student_logits, y)

        # 总损失
        total_loss = alpha*ce_loss + beta*kd_loss + gamma*feature_loss
        return total_loss
```

---

#### Self-Distillation (自蒸馏)

**定义**：模型自己教自己。

**方法**：

**1. Born-Again Networks**
```python
# 阶段1：训练第一代模型
model_gen1 = train(data)

# 阶段2：用第一代教第二代(相同架构)
model_gen2 = distill(model_gen1, data)

# 可选：迭代多代
model_gen3 = distill(model_gen2, data)
```

**2. Deep Mutual Learning**
```python
# 多个模型互相学习
models = [Model() for _ in range(num_models)]

for x, y in data:
    losses = []
    for i, model_i in enumerate(models):
        # 自己的损失
        pred_i = model_i(x)
        ce_loss = F.cross_entropy(pred_i, y)

        # 向其他模型学习
        kd_loss = 0
        for j, model_j in enumerate(models):
            if i != j:
                with torch.no_grad():
                    pred_j = model_j(x)
                kd_loss += KL_div(pred_i, pred_j)

        total_loss = ce_loss + kd_loss
        losses.append(total_loss)

    # 同时更新所有模型
    for loss, model in zip(losses, models):
        loss.backward()
        model.optimizer.step()
```

---

#### Online Distillation (在线蒸馏)

**定义**：Teacher和Student同时训练。

```python
# Teacher和Student同时从数据学习
for x, y in data:
    # Teacher更新
    teacher_loss = F.cross_entropy(teacher(x), y)
    teacher_loss.backward()
    teacher_optimizer.step()

    # Student从Teacher学习
    with torch.no_grad():
        teacher_pred = teacher(x)

    student_pred = student(x)
    student_loss = (
        alpha * F.cross_entropy(student_pred, y) +
        (1-alpha) * KL_div(student_pred, teacher_pred)
    )
    student_loss.backward()
    student_optimizer.step()
```

---

### 18.2 剪枝进阶

#### Lottery Ticket Hypothesis (彩票假说)

**定义**：大网络中存在小子网络(中奖彩票)，单独训练也能达到相似性能。

**发现**：
```python
# 1. 随机初始化
weights_0 = random_init()

# 2. 训练网络
weights_trained = train(weights_0)

# 3. 根据重要性剪枝
mask = find_important_weights(weights_trained)  # 保留5-20%

# 4. 关键：用原始初始化训练剪枝后的网络
winning_ticket = mask * weights_0  # 重置到初始值
final_weights = train(winning_ticket)

# 结果：winning_ticket性能接近完整网络!
```

**影响**：
- 理论意义重大
- 预训练可能在寻找好的子结构

---

#### Magnitude Pruning (幅度剪枝)

**定义**：移除权重绝对值小的连接。

```python
def magnitude_prune(model, sparsity=0.5):
    """
    sparsity: 要剪掉的比例
    """
    # 收集所有权重
    all_weights = torch.cat([
        param.flatten()
        for param in model.parameters()
    ])

    # 计算阈值
    threshold = torch.quantile(torch.abs(all_weights), sparsity)

    # 应用mask
    for param in model.parameters():
        mask = torch.abs(param) > threshold
        param.data *= mask

    return model
```

---

#### Movement Pruning

**定义**：剪掉训练中移向零的权重。

```python
def movement_pruning(model, optimizer, sparsity):
    """
    跟踪权重的'移动方向'
    """
    # 在训练循环中
    for step in range(training_steps):
        loss = compute_loss(model(x), y)
        loss.backward()

        # 计算importance score
        for param in model.parameters():
            # 权重和梯度方向相反 → 向零移动 → 不重要
            importance = -param * param.grad
            # 存储importance

        optimizer.step()

    # 根据importance剪枝
    threshold = compute_threshold(importance_scores, sparsity)
    apply_mask(model, importance_scores > threshold)
```

---

### 18.3 神经架构搜索进阶

#### Once-for-All (OFA) Networks

**定义**：训练一个超网络，支持多种架构配置。

```python
class OFANetwork:
    def __init__(self):
        # 支持多种深度、宽度、kernel size
        self.layers = nn.ModuleList([
            ElasticLayer(
                depths=[2, 3, 4],        # 可选深度
                widths=[128, 192, 256],  # 可选宽度
                kernels=[3, 5, 7]        # 可选kernel
            )
            for _ in range(20)
        ])

    def sample_subnet(self):
        # 采样一个子网络配置
        config = {
            'depth': random.choice([2, 3, 4]),
            'width': random.choice([128, 192, 256]),
            'kernel': random.choice([3, 5, 7]),
        }
        return config

    def forward(self, x, config):
        # 根据配置执行
        for layer in self.layers[:config['depth']]:
            x = layer(x, width=config['width'], kernel=config['kernel'])
        return x

# 训练：渐进式收缩
# 1. 训练最大网络
# 2. 逐步支持更小的子网络
# 3. 最终一个网络支持所有配置

# 部署：选择符合资源限制的配置
if device == "mobile":
    config = {'depth': 2, 'width': 128, 'kernel': 3}
elif device == "server":
    config = {'depth': 4, 'width': 256, 'kernel': 7}
```

---

#### Hardware-Aware NAS

**定义**：考虑硬件约束的NAS。

```python
def hardware_aware_search(search_space, target_device):
    best_arch = None
    best_score = 0

    for arch in search_space:
        # 评估准确率
        accuracy = evaluate_accuracy(arch)

        # 评估延迟(在目标设备上实测)
        latency = measure_latency(arch, target_device)

        # 评估能耗
        energy = measure_energy(arch, target_device)

        # 多目标评分
        if latency < latency_constraint and energy < energy_budget:
            score = accuracy / (latency * energy)
            if score > best_score:
                best_score = score
                best_arch = arch

    return best_arch
```

**考虑因素**：
- 延迟(latency)
- 吞吐量(throughput)
- 内存占用
- 能耗
- 特定硬件优化(如GPU tensor cores)

---

## 结语

这份完整的AI大模型专有名词指南现已覆盖**1000+核心术语**，从基础架构到前沿技术，从训练优化到实际应用，包括：

- **基础理论**：Transformer、Attention、优化器、损失函数
- **训练技术**：分布式训练、混合精度、梯度累积、学习率调度
- **推理优化**：量化、剪枝、蒸馏、KV Cache、Flash Attention
- **高级方法**：RLHF、DPO、RAG、CoT、元学习、持续学习
- **前沿研究**：State Space Models、神经符号AI、可解释性、鲁棒性
- **应用场景**：多模态、代码生成、领域模型、多语言NLP

持续更新中，欢迎补充！🚀
