# Transformer 架构:注意力机制的数学本质

## 1. 引言

Transformer 彻底改变了深度学习,从 NLP(BERT, GPT)到 CV(ViT)再到多模态(CLIP),无处不在。它的核心创新是**自注意力机制(Self-Attention)**,抛弃了 RNN 和 CNN,实现了真正的并行化。本教程从数学原理到代码实现,完整解析 Transformer。

---

## 2. 序列建模的历史演进

### 2.1 RNN 的困境

**递归结构:**
```
h_t = f(h_(t-1), x_t)
```

**问题:**
1. **无法并行:** h_t 依赖 h_(t-1),必须顺序计算
2. **长程依赖:** 梯度消失/爆炸(即使用 LSTM/GRU)
3. **信息瓶颈:** 所有信息压缩到固定维度的 h_t

### 2.2 CNN 的局限

**卷积核局部性:**
```
感受野有限,需多层堆叠才能捕获长距离依赖
```

### 2.3 Attention 的革命

**核心思想:** 直接建模任意位置之间的依赖关系
```
output_i = Σ attention(i, j) · value_j
           j
```

**优势:**
- ✓ 完全并行化
- ✓ 常数级路径长度(任意两位置)
- ✓ 动态权重(依赖输入)

---

## 3. 注意力机制的数学基础

### 3.1 注意力的直观理解

**问题:** 给定查询(Query),从一堆键值对(Key-Value)中检索相关信息。

**类比数据库:**
```sql
SELECT value FROM table WHERE similarity(query, key) IS HIGH
```

**软检索:** 不是硬选择一个,而是加权平均所有值。

---

### 3.2 缩放点积注意力(Scaled Dot-Product Attention)

**输入:**
- Query: Q ∈ ℝ^(n × d_k)
- Key: K ∈ ℝ^(m × d_k)
- Value: V ∈ ℝ^(m × d_v)

**计算步骤:**

**(1) 计算相似度(点积)**
```
S = Q · K^T  ∈ ℝ^(n × m)

S_ij = q_i · k_j = Σ q_i[l] · k_j[l]
                   l
```

**为什么点积?**
- 计算高效(矩阵乘法)
- 几何意义:余弦相似度(归一化后)

**(2) 缩放**
```
S_scaled = S / √d_k
```

**为什么缩放?**

点积方差随维度增长:
```
Var[q·k] = d_k · Var[q] · Var[k]
```

当 d_k 很大时,点积值很大,softmax 梯度趋近 0(饱和)。

除以 √d_k 使方差稳定:
```
Var[q·k / √d_k] = Var[q] · Var[k]
```

**(3) Softmax 归一化**
```
A = softmax(S_scaled)

A_ij = exp(S_ij / √d_k) / Σ exp(S_ik / √d_k)
                           k
```

**性质:**
- Σ A_ij = 1 (每行和为 1)
- A_ij ∈ [0, 1] (概率分布)

**(4) 加权求和**
```
Output = A · V  ∈ ℝ^(n × d_v)

output_i = Σ A_ij · v_j
           j
```

**完整公式:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

---

### 3.3 代码实现

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n, d_k)
    K: (batch, m, d_k)
    V: (batch, m, d_v)
    """
    d_k = Q.shape[-1]

    # 计算注意力分数
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # (batch, n, m)

    # 应用 mask (可选)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = softmax(scores, axis=-1)  # (batch, n, m)

    # 加权求和
    output = attention_weights @ V  # (batch, n, d_v)

    return output, attention_weights

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

---

### 3.4 Mask 的作用

**(1) Padding Mask**
```
处理变长序列,忽略 padding 位置

mask_ij = 1 if j 是真实 token
          0 if j 是 padding
```

**(2) Causal Mask (因果 mask)**
```
防止看到未来信息(用于自回归生成)

mask_ij = 1 if j ≤ i
          0 if j > i

对应上三角矩阵
```

**代码:**
```python
def create_causal_mask(seq_len):
    """下三角矩阵"""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

# 示例: seq_len = 4
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

---

## 4. 多头注意力(Multi-Head Attention)

### 4.1 动机

单个注意力头的局限:
- 只能关注一种模式
- 类比:单个卷积核 vs 多个卷积核

**解决方案:** 多个注意力头并行,捕获不同的依赖关系。

---

### 4.2 数学定义

**参数:**
```
W^Q_h ∈ ℝ^(d_model × d_k)    # 每个头的 Q 投影矩阵
W^K_h ∈ ℝ^(d_model × d_k)    # 每个头的 K 投影矩阵
W^V_h ∈ ℝ^(d_model × d_v)    # 每个头的 V 投影矩阵
W^O ∈ ℝ^(h·d_v × d_model)     # 输出投影矩阵
```

**计算流程:**

**(1) 线性投影(每个头)**
```
Q_h = X · W^Q_h  ∈ ℝ^(n × d_k)
K_h = X · W^K_h  ∈ ℝ^(n × d_k)
V_h = X · W^V_h  ∈ ℝ^(n × d_v)
```

**(2) 独立计算注意力**
```
head_h = Attention(Q_h, K_h, V_h)  ∈ ℝ^(n × d_v)
```

**(3) 拼接所有头**
```
Concat = [head_1; head_2; ...; head_h]  ∈ ℝ^(n × h·d_v)
```

**(4) 输出投影**
```
Output = Concat · W^O  ∈ ℝ^(n × d_model)
```

**标准配置:**
```
d_model = 512
h = 8
d_k = d_v = d_model / h = 64
```

---

### 4.3 代码实现

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 投影矩阵
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x, batch_size):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, num_heads, seq_len, d_k)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, X, mask=None):
        """
        X: (batch, seq_len, d_model)
        """
        batch_size = X.shape[0]

        # 线性投影
        Q = X @ self.W_Q  # (batch, seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # 分割成多头
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 注意力计算
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )  # (batch, num_heads, seq_len, d_k)

        # 拼接多头
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, d_k)
        concat = attention_output.reshape(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)

        # 输出投影
        output = concat @ self.W_O

        return output, attention_weights
```

---

### 4.4 多头的可解释性

**不同头关注不同模式:**
- Head 1: 句法关系(主谓宾)
- Head 2: 共指消解(代词与先行词)
- Head 3: 位置关系(相邻词)

**可视化:**
```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    """
    attention_weights: (num_heads, seq_len, seq_len)
    """
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(15, 3))

    for i in range(num_heads):
        ax = axes[i]
        ax.imshow(attention_weights[i], cmap='viridis')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        ax.set_title(f'Head {i+1}')

    plt.tight_layout()
    plt.show()
```

---

## 5. 位置编码(Positional Encoding)

### 5.1 问题

**注意力的排列不变性:**
```
Attention({x_1, x_2, x_3}) = Attention({x_2, x_1, x_3})
```

打乱输入顺序,输出不变!

**但序列任务需要位置信息:**
- "狗咬人" vs "人咬狗"

---

### 5.2 绝对位置编码

**Sinusoidal Encoding (Transformer 原论文):**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**参数:**
- pos: 位置索引 (0, 1, 2, ...)
- i: 维度索引 (0, 1, ..., d_model/2 - 1)

**特性:**
1. **确定性:** 不需要学习
2. **外推性:** 可处理训练时未见过的长度
3. **相对位置:** PE(pos+k) 可表示为 PE(pos) 的线性组合

**代码:**
```python
def positional_encoding(seq_len, d_model):
    """
    seq_len: 序列长度
    d_model: 模型维度
    """
    PE = np.zeros((seq_len, d_model))

    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# 使用
PE = positional_encoding(100, 512)
X_with_pos = X + PE  # 加到输入上
```

**为什么能编码相对位置?**

三角恒等式:
```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
```

因此 PE(pos+k) 可以表示为 PE(pos) 的线性变换。

---

### 5.3 可学习位置编码

**简单方法:** 直接学习位置嵌入
```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
```

**优势:** 灵活,表达力强
**劣势:** 无法外推到更长序列

**应用:** BERT, GPT

---

### 5.4 相对位置编码

**动机:** 直接建模相对距离 (i-j)

**方法:**
修改注意力计算,加入相对位置偏置:
```
A_ij = softmax((q_i·k_j + R(i-j)) / √d_k)
```

**应用:** Transformer-XL, T5

---

## 6. Transformer 编码器(Encoder)

### 6.1 单层结构

```
Input
  ↓
Multi-Head Attention
  ↓
Add & Norm (残差连接 + LayerNorm)
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```

---

### 6.2 组件详解

**(1) Multi-Head Self-Attention**
```
Q = K = V = X  (自注意力)
```

**(2) 残差连接 + LayerNorm**
```
X_out = LayerNorm(X + Sublayer(X))
```

**为什么有效?**
- 残差:缓解梯度消失,允许堆叠深层
- LayerNorm:稳定训练

**(3) Feed-Forward Network**
```
FFN(x) = max(0, x·W1 + b1)·W2 + b2
```

**配置:**
```
d_ff = 4 × d_model = 2048
```

**逐位置(Position-wise):** 每个位置独立应用相同 FFN

---

### 6.3 完整代码

```python
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output, _ = self.mha.forward(x, mask)
        attn_output = dropout(attn_output, self.dropout)
        x = self.layernorm1(x + attn_output)  # Add & Norm

        # Feed-Forward
        ffn_output = self.ffn.forward(x)
        ffn_output = dropout(ffn_output, self.dropout)
        x = self.layernorm2(x + ffn_output)  # Add & Norm

        return x

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

### 6.4 堆叠多层

```python
class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x
```

**标准配置:**
- BERT-Base: 12 层
- BERT-Large: 24 层
- GPT-3: 96 层

---

## 7. Transformer 解码器(Decoder)

### 7.1 结构差异

与编码器的区别:
1. **Masked Self-Attention:** 防止看到未来
2. **Cross-Attention:** 关注编码器输出

```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention (Q 来自解码器,K/V 来自编码器)
  ↓
Add & Norm
  ↓
Feed-Forward
  ↓
Add & Norm
  ↓
Output
```

---

### 7.2 Cross-Attention

**数学:**
```
Q = Decoder_Output
K = V = Encoder_Output

CrossAttn = Attention(Q, K, V)
```

**直观理解:** 解码器查询编码器的相关信息

**应用:** 机器翻译(源语言 → 目标语言)

---

## 8. 实战:从零构建 Transformer

### 8.1 完整模型

```python
class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
        # 嵌入层
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = positional_encoding(5000, d_model)

        # 编码器和解码器
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # 输出层
        self.fc_out = np.random.randn(d_model, tgt_vocab_size) * 0.01

    def encode(self, src, src_mask):
        src_emb = self.src_embedding(src) + self.pos_encoding[:src.shape[1]]
        return self.encoder.forward(src_emb, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding[:tgt.shape[1]]
        return self.decoder.forward(tgt_emb, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = decoder_output @ self.fc_out
        return logits
```

---

### 8.2 训练技巧

**(1) Teacher Forcing**
```
训练时:输入真实目标序列的前缀
推理时:输入模型自己生成的序列
```

**(2) Label Smoothing**
```
将硬标签 [0, 0, 1, 0] 软化为 [ε, ε, 1-3ε, ε]
```

**效果:** 防止过拟合,提升泛化

**(3) Warmup + Cosine Decay**
```python
def lr_schedule(step, d_model, warmup_steps=4000):
    arg1 = 1 / np.sqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
    return (1 / np.sqrt(d_model)) * min(arg1, arg2)
```

---

## 9. Transformer 变体

### 9.1 BERT (Bidirectional Encoder)

**结构:** 只用编码器

**预训练任务:**
1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP)

**应用:** 文本分类,问答,命名实体识别

---

### 9.2 GPT (Autoregressive Decoder)

**结构:** 只用解码器(带 causal mask)

**预训练任务:** 语言建模(预测下一个词)

**应用:** 文本生成,少样本学习

---

### 9.3 Vision Transformer (ViT)

**核心思想:** 将图像分成 patches,当作序列

```
Image (224×224×3)
  ↓ 分割成 16×16 patches
Sequence of 196 patches
  ↓ 线性投影
Patch Embeddings (196, 768)
  ↓ + 位置编码
Transformer Encoder
  ↓
分类头
```

**突破:** 证明 Transformer 在 CV 也有效

---

## 10. 理论洞察

### 10.1 计算复杂度

**自注意力:**
```
O(n² · d)
```
- n: 序列长度
- d: 模型维度

**问题:** 对长序列不友好

**解决方案:**
- Sparse Attention (Longformer)
- Linear Attention (Performer)
- Hierarchical Attention (Reformer)

---

### 10.2 表达能力

**定理:** Transformer 是图灵完备的(在深度足够时)

**证明思路:**
- 注意力可模拟条件分支
- FFN 可模拟计算
- 多层堆叠可实现任意逻辑

---

### 10.3 归纳偏置(Inductive Bias)

| 模型 | 归纳偏置 |
|------|---------|
| CNN | 局部性,平移不变性 |
| RNN | 顺序性,马尔可夫性 |
| Transformer | 几乎无(全靠数据) |

**结论:** Transformer 需要更多数据,但泛化能力更强

---

## 11. 总结

### 核心概念
- **Self-Attention:** 建模全局依赖
- **多头机制:** 捕获多种模式
- **位置编码:** 注入顺序信息
- **残差 + LayerNorm:** 稳定深层训练

### 关键优势
1. 完全并行化(训练快)
2. 长程依赖建模好
3. 可解释性强

### 实践建议
- 小数据:用预训练模型(BERT, GPT)
- 长序列:用高效变体(Longformer)
- 多模态:用统一框架(ViT, CLIP)

### 前沿方向
- 高效 Transformer (Flashattention)
- 超长上下文(100K+ tokens)
- 多模态统一架构

**记住:** Attention is All You Need!
