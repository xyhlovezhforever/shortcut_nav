# 多模态AI评估技术详解

## 一、概述

多模态AI涉及图像、文本、视频、音频等多种模态的理解和生成。评估多模态模型需要专门设计的指标，既要考虑单模态质量，也要评估跨模态对齐。本文档详细介绍图像描述、视觉问答、文本到图像生成等任务的评估方法。

---

## 二、图像描述生成评估

### 2.1 BLEU（Bilingual Evaluation Understudy）

#### 2.1.1 原理

**定义**：计算生成文本和参考文本之间的N-gram精确匹配。

**公式**：
```
BLEU = BP × exp(Σ(w_n × log(p_n)))

其中：
- p_n: n-gram精确率
- w_n: 权重（通常均等，如BLEU-4每个1/4）
- BP: 简短惩罚（Brevity Penalty）
```

**简短惩罚**：
```
BP = {
    1,                  if c > r
    exp(1 - r/c),      if c ≤ r
}

c: 候选长度
r: 参考长度
```

#### 2.1.2 计算示例

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# 单句BLEU
reference = [['a', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['a', 'cat', 'on', 'the', 'mat']

# BLEU-1, BLEU-2, BLEU-3, BLEU-4
bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-4: {bleu4:.4f}")

# 语料库级别BLEU
references = [
    [['a', 'cat', 'on', 'the', 'mat']],
    [['a', 'dog', 'in', 'the', 'park']],
]
candidates = [
    ['a', 'cat', 'on', 'mat'],
    ['a', 'dog', 'in', 'park'],
]

corpus_bleu_score = corpus_bleu(references, candidates)
print(f"Corpus BLEU: {corpus_bleu_score:.4f}")
```

#### 2.1.3 优缺点

**优点**：
- 简单、快速
- 广泛使用，可比较
- 考虑词序

**缺点**：
- 只关注精确匹配，忽略同义词
- 对短句子不友好
- 不考虑语义相似性

---

### 2.2 METEOR（Metric for Evaluation of Translation with Explicit ORdering）

#### 2.2.1 改进点

**相比BLEU的优势**：
1. 考虑同义词匹配
2. 词干匹配
3. 释义匹配
4. 兼顾精确率和召回率

**对齐方式**：
- 精确匹配（Exact）
- 词干匹配（Stem）
- 同义词匹配（Synonym）

#### 2.2.2 计算公式

```
METEOR = (1 - Penalty) × F_mean

F_mean = 10 × P × R / (R + 9P)

Penalty = 0.5 × (chunks / matches)^3

其中：
- P: 精确率
- R: 召回率
- chunks: 匹配块数量
```

#### 2.2.3 使用示例

```python
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download('wordnet')

reference = "a cat is sitting on the mat"
candidate = "a cat sits on a mat"

score = meteor_score([reference.split()], candidate.split())
print(f"METEOR: {score:.4f}")

# 支持同义词
reference2 = "a beautiful sunset over the ocean"
candidate2 = "a gorgeous sunset above the sea"
score2 = meteor_score([reference2.split()], candidate2.split())
print(f"METEOR (with synonyms): {score2:.4f}")
```

---

### 2.3 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

#### 2.3.1 变体

**ROUGE-N**：N-gram召回率
```python
from rouge import Rouge

rouge = Rouge()

reference = "a cat is sitting on the mat"
candidate = "a cat on the mat"

scores = rouge.get_scores(candidate, reference)[0]

print(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
```

**ROUGE-L**：最长公共子序列（LCS）
```
LCS("a cat on mat", "a cat is on the mat") = "a cat on mat"
```

**ROUGE-W**：加权LCS（连续匹配权重更高）

---

### 2.4 CIDEr（Consensus-based Image Description Evaluation）

#### 2.4.1 专为图像描述设计

**核心思想**：
- 基于TF-IDF加权N-gram相似度
- 考虑多个参考描述之间的共识
- 常见词权重低，罕见词权重高

**公式**：
```
CIDEr = (1/m) × Σ CIDEr_n(c, S)

CIDEr_n = (1/|S|) × Σ cos(g_n(c), g_n(s_j))

其中：
- g_n: TF-IDF加权的n-gram向量
- c: 候选描述
- S: 参考描述集合
- m: n的数量（通常1-4）
```

#### 2.4.2 使用示例

```python
from pycocoevalcap.cider.cider import Cider

# 准备数据格式
gts = {
    0: ['a cat sitting on a mat', 'cat on the mat', 'a cat is resting on a mat']
}
res = {
    0: ['a cat on a mat']
}

cider_scorer = Cider()
score, scores = cider_scorer.compute_score(gts, res)

print(f"CIDEr: {score:.4f}")
```

#### 2.4.3 优势

- 与人类判断相关性高（0.9+）
- 重视描述性词汇
- 考虑多样性

---

### 2.5 SPICE（Semantic Propositional Image Caption Evaluation）

#### 2.5.1 基于场景图的语义评估

**评估维度**：
1. **对象（Objects）**：图像中的实体
2. **属性（Attributes）**：对象的特征
3. **关系（Relations）**：对象之间的关系

**场景图示例**：
```
图像描述："A brown dog is playing with a red ball in the park"

场景图：
- Objects: dog, ball, park
- Attributes: dog(brown), ball(red)
- Relations: playing_with(dog, ball), in(dog, park)
```

#### 2.5.2 计算流程

```
1. 解析候选和参考描述 → 场景图
2. 比较场景图的元组（对象、属性、关系）
3. 计算F-score
```

#### 2.5.3 使用示例

```python
from pycocoevalcap.spice.spice import Spice

gts = {
    0: ['a brown dog playing with a red ball']
}
res = {
    0: ['a dog playing with a ball']
}

spice_scorer = Spice()
score, scores = spice_scorer.compute_score(gts, res)

print(f"SPICE: {score:.4f}")
print(f"Details: {scores[0]}")
# 输出：对象匹配度、属性匹配度、关系匹配度
```

#### 2.5.4 优势

- 捕捉语义层面的准确性
- 不依赖表面词汇匹配
- 对幻觉（hallucination）敏感

---

## 三、图像-文本对齐评估

### 3.1 CLIPScore

#### 3.1.1 原理

**定义**：使用CLIP模型计算图像和文本的语义相似度。

**CLIP（Contrastive Language-Image Pre-training）**：
- 在4亿图像-文本对上训练
- 图像和文本映射到同一嵌入空间
- 通过对比学习对齐

#### 3.1.2 计算方法

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像和描述
image = Image.open("cat.jpg")
texts = [
    "a cat sitting on a mat",
    "a dog running in a park",
    "a beautiful sunset"
]

# 计算相似度
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# 图像-文本相似度
logits_per_image = outputs.logits_per_image  # [1, 3]
probs = logits_per_image.softmax(dim=1)  # 归一化

print("CLIPScores:")
for i, text in enumerate(texts):
    print(f"  {text:40} | Score: {logits_per_image[0, i].item():.4f} | Prob: {probs[0, i].item():.4f}")

# 输出示例：
# a cat sitting on a mat                   | Score: 28.7234 | Prob: 0.9823
# a dog running in a park                  | Score: 18.4521 | Prob: 0.0156
# a beautiful sunset                       | Score: 15.2341 | Prob: 0.0021
```

#### 3.1.3 图像描述评估

```python
from torchmetrics.multimodal import CLIPScore

clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")

# 批量评估
images = [image1, image2, image3]  # PIL Images
captions = ["caption1", "caption2", "caption3"]

score = clip_score(images, captions)
print(f"Average CLIPScore: {score.item():.4f}")
```

#### 3.1.4 应用场景

- 图像描述生成评估
- 文本到图像生成评估（DALL-E、Stable Diffusion等）
- 图像检索
- 多模态对齐质量检测

---

### 3.2 ImageReward

#### 3.2.1 基于人类偏好的评估

**特点**：
- 训练于人类偏好数据
- 更符合人类审美
- 专为文生图模型设计

```python
import ImageReward as RM

model = RM.load("ImageReward-v1.0")

# 评估文生图结果
image_path = "generated_image.jpg"
prompt = "a futuristic city with flying cars"

score = model.score(prompt, image_path)
print(f"ImageReward Score: {score:.4f}")
# 高分表示更符合人类偏好
```

---

## 四、视觉问答（VQA）评估

### 4.1 VQA准确率

#### 4.1.1 标准VQA指标

**定义**：考虑多个人类标注答案的一致性。

**公式**：
```
Accuracy = min(
    (匹配答案的人数) / 3,
    1
)
```

**示例**：
```
问题："图中有几只猫？"
参考答案：["2", "2", "2", "two", "2"]  （5个标注者）

候选答案："2"
匹配数：4
准确率：min(4/3, 1) = 1.0

候选答案："two"
匹配数：1
准确率：min(1/3, 1) = 0.33
```

#### 4.1.2 实现

```python
def vqa_accuracy(predicted_answer, ground_truth_answers):
    """
    predicted_answer: 预测答案（字符串）
    ground_truth_answers: 参考答案列表
    """
    num_matches = sum([1 for gt in ground_truth_answers if gt == predicted_answer])
    accuracy = min(num_matches / 3.0, 1.0)
    return accuracy

# 示例
gt_answers = ["2", "2", "2", "two", "2"]
pred = "2"

acc = vqa_accuracy(pred, gt_answers)
print(f"VQA Accuracy: {acc:.4f}")
```

---

### 4.2 开放式VQA评估

**挑战**：答案可能有多种表述方式

**解决方案**：
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity_score(predicted, references):
    """基于语义相似度的评估"""
    pred_emb = model.encode(predicted)
    ref_embs = model.encode(references)

    similarities = util.cos_sim(pred_emb, ref_embs)
    max_sim = similarities.max().item()

    return max_sim

# 示例
predicted = "It's a cat"
references = ["cat", "a cat", "feline", "kitten"]

score = semantic_similarity_score(predicted, references)
print(f"Semantic Similarity: {score:.4f}")
```

---

## 五、文本到图像生成评估

### 5.1 FID（Fréchet Inception Distance）

#### 5.1.1 原理

**定义**：衡量生成图像和真实图像在特征空间中的分布距离。

**步骤**：
```
1. 使用Inception-v3提取图像特征
2. 计算真实图像特征的均值μ_r和协方差Σ_r
3. 计算生成图像特征的均值μ_g和协方差Σ_g
4. 计算Fréchet距离
```

**公式**：
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r × Σ_g))
```

#### 5.1.2 使用示例

```python
from pytorch_fid import fid_score

# 计算FID
fid_value = fid_score.calculate_fid_given_paths(
    paths=['path/to/real_images', 'path/to/generated_images'],
    batch_size=50,
    device='cuda',
    dims=2048  # Inception-v3特征维度
)

print(f"FID: {fid_value:.2f}")
# 越低越好（< 10: 优秀, 10-30: 良好, > 50: 较差）
```

---

### 5.2 IS（Inception Score）

#### 5.2.1 原理

**评估维度**：
1. **质量**：每张图像的类别预测应该清晰（低熵）
2. **多样性**：整体类别分布应该均匀（高熵）

**公式**：
```
IS = exp(E_x[KL(p(y|x) || p(y))])

其中：
- p(y|x): 单张图像的类别分布
- p(y): 所有图像的边缘类别分布
```

#### 5.2.2 实现

```python
import torch
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np

def calculate_inception_score(images, batch_size=32, splits=10):
    """
    images: torch.Tensor [N, 3, H, W]
    """
    # 加载Inception模型
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    preds = []

    # 批量预测
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            pred = inception_model(batch)
        preds.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 计算IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# 示例
# generated_images = torch.randn(1000, 3, 299, 299)  # 假设数据
# is_mean, is_std = calculate_inception_score(generated_images)
# print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
```

---

### 5.3 人类评估

#### 5.3.1 评估维度

**图像质量**：
- 清晰度
- 真实感
- 美学

**文本-图像一致性**：
- 对象匹配
- 属性匹配
- 关系匹配
- 整体语义

#### 5.3.2 评估协议

```python
import random

def human_evaluation_interface(images, prompts):
    """人类评估界面示例"""
    results = []

    for img, prompt in zip(images, prompts):
        print(f"\nPrompt: {prompt}")
        # 显示图像
        display(img)

        # 收集评分
        quality = int(input("图像质量 (1-5): "))
        alignment = int(input("与提示词一致性 (1-5): "))
        aesthetics = int(input("美学 (1-5): "))

        results.append({
            'quality': quality,
            'alignment': alignment,
            'aesthetics': aesthetics
        })

    return results
```

---

## 六、综合评估框架

### 6.1 CLIP-based评估套件

```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class MultimodalEvaluator:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def clip_score(self, images, captions):
        """图像-文本对齐分数"""
        inputs = self.clip_processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        logits_per_image = outputs.logits_per_image
        return logits_per_image.diagonal().mean().item()

    def caption_similarity(self, generated_captions, reference_captions):
        """描述相似度（CLIP文本空间）"""
        gen_inputs = self.clip_processor(text=generated_captions, return_tensors="pt", padding=True)
        ref_inputs = self.clip_processor(text=reference_captions, return_tensors="pt", padding=True)

        with torch.no_grad():
            gen_embs = self.clip_model.get_text_features(**gen_inputs)
            ref_embs = self.clip_model.get_text_features(**ref_inputs)

        # 归一化
        gen_embs = gen_embs / gen_embs.norm(dim=-1, keepdim=True)
        ref_embs = ref_embs / ref_embs.norm(dim=-1, keepdim=True)

        # 余弦相似度
        similarities = (gen_embs * ref_embs).sum(dim=-1)
        return similarities.mean().item()

    def evaluate_image_captioning(self, images, generated_captions, reference_captions):
        """综合评估图像描述"""
        results = {
            'clip_score': self.clip_score(images, generated_captions),
            'caption_similarity': self.caption_similarity(generated_captions, reference_captions),
        }

        # 可以添加其他指标
        # results['bleu'] = compute_bleu(generated_captions, reference_captions)
        # results['cider'] = compute_cider(generated_captions, reference_captions)

        return results

# 使用示例
evaluator = MultimodalEvaluator()

images = [Image.open("cat.jpg")]
generated = ["a cat sitting on a mat"]
references = ["a cat on the mat"]

scores = evaluator.evaluate_image_captioning(images, generated, references)
print(scores)
```

---

## 七、评估指标对比总结

### 7.1 图像描述生成指标

| 指标 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **BLEU** | 快速、简单 | 忽略同义词、语义 | 快速baseline |
| **METEOR** | 考虑同义词 | 计算较慢 | 通用评估 |
| **ROUGE** | 关注召回 | 适合摘要，不适合描述 | 摘要任务 |
| **CIDEr** | 与人类高度相关 | 需要多个参考 | **推荐** |
| **SPICE** | 语义准确性 | 计算昂贵 | 精细评估 |
| **CLIPScore** | 跨模态对齐 | 依赖CLIP质量 | **现代推荐** |

### 7.2 文生图评估指标

| 指标 | 评估内容 | 优点 | 缺点 |
|------|---------|------|------|
| **FID** | 分布距离 | 全面、客观 | 需要大量样本 |
| **IS** | 质量+多样性 | 单一数值 | 偏向ImageNet类别 |
| **CLIPScore** | 文本对齐 | 直接、准确 | **推荐** |
| **人类评估** | 主观质量 | 最准确 | 昂贵、耗时 |

---

## 八、最佳实践

### 8.1 评估策略

**图像描述生成**：
```
1. 自动指标：CIDEr + CLIPScore
2. 语义准确性：SPICE
3. 人类评估：抽样评估（10-20%）
```

**文本到图像**：
```
1. 质量：FID
2. 对齐：CLIPScore
3. 多样性：IS
4. 人类评估：美学+一致性
```

**视觉问答**：
```
1. 准确率：VQA-Accuracy
2. 语义匹配：BERTScore / Semantic Similarity
3. 人类评估：复杂推理案例
```

### 8.2 评估代码模板

```python
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def comprehensive_evaluation(annotations_file, results_file):
    """
    annotations_file: COCO格式的标注文件
    results_file: 模型生成结果文件
    """
    coco = COCO(annotations_file)
    coco_res = coco.loadRes(results_file)

    eval = COCOEvalCap(coco, coco_res)
    eval.evaluate()

    # 打印所有指标
    print("Evaluation Results:")
    for metric, score in eval.eval.items():
        print(f"{metric:10}: {score:.4f}")

    return eval.eval

# 示例输出：
# BLEU-1    : 0.7234
# BLEU-4    : 0.3421
# METEOR    : 0.2678
# ROUGE_L   : 0.5432
# CIDEr     : 1.0234
# SPICE     : 0.2145
```

---

## 九、总结

**关键要点**：

1. **自动指标不是万能的**：
   - 必须结合人类评估
   - 不同指标衡量不同方面
   - 多指标综合判断

2. **推荐组合**：
   - 图像描述：**CIDEr + CLIPScore + 人类评估**
   - 文生图：**FID + CLIPScore + ImageReward**
   - VQA：**VQA-Accuracy + 语义相似度**

3. **现代趋势**：
   - 基于大模型的评估（如CLIP）
   - 人类偏好学习（如ImageReward）
   - 端到端可微分指标

4. **实践建议**：
   - 早期开发：快速自动指标（BLEU、CLIPScore）
   - 模型优化：全面自动评估
   - 最终验证：人类评估+自动指标

多模态评估是一个复杂的领域，选择合适的指标组合对于准确衡量模型性能至关重要！
