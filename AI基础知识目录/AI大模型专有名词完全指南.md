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
