---
date: 2026-04-17
title: "训练 Embedding 与微调表示模型：从 SBERT 对比学习到 SetFit 少样本"
description: "想给自己领域做一个高质量 Embedding 模型？手里只有几十条标注数据还能微调出准确的分类器？这两个问题的答案都在 Sentence-Transformer 生态里——对比学习、Triplet Loss、SetFit、TSDAE。"
tags: [SBERT, Sentence-Transformers, 对比学习, SetFit, Fine-tuning, NER]
---

# 训练 Embedding 与微调表示模型：从 SBERT 对比学习到 SetFit 少样本

> 本文对应原书 **第 10 章 Creating Text Embedding Models** 和 **第 11 章 Fine-Tuning Representation Models for Classification**，覆盖：SBERT 架构、对比学习与三种主流 Loss、TSDAE 无监督训练、BERT 监督分类微调、冻结层技巧、SetFit 少样本方案、命名实体识别。

## 开篇两问

这两章其实是在回答工程实践中最常见的两个问题：

1. 我有一堆自有语料（法律条款、医疗记录、企业内部文档），通用 Embedding 模型在这些数据上效果一般，怎么做一个**自家的 Embedding 模型**？
2. 我的分类任务冷启动——每类只有 8 条、16 条标注，连 BERT 微调都喂不饱，还能不能训？

两个答案都落在同一个生态里：**sentence-transformers + 对比学习**。第 10 章讲怎么从头训/微调一个 Embedding 模型，第 11 章讲怎么把 Embedding 模型/BERT 加上分类头做下游任务——SetFit 则是把二者融合在一起的少样本神器。

## 第一部分：为什么 BERT 的 CLS 不能直接当句向量？

### BERT 句向量的两条老路

把 BERT 当 Embedding 模型用，最直觉的两种方式是：

- **CLS token pooling**：把 `[CLS]` 位置的输出向量当句向量
- **Mean pooling**：把所有 token 的输出平均一下

看似合理，但实测里这两种向量做语义相似度，效果**甚至不如平均 GloVe 词向量**。原因是：原始 BERT 的训练目标（MLM + NSP）从来没有显式优化过"整个句子的向量要能表征语义"。CLS 学到的是"这两句是不是相邻"，Mean pooling 只是把 token 向量取均值，没有保证余弦距离对应语义距离。

### Cross-Encoder 也不是好选择

另一条路是 **Cross-Encoder**：把两句拼起来（用 `<SEP>` 分隔）送进 BERT，加一个分类头输出相似度分数。

准确度很高，但**计算开销爆炸**——如果要在 10,000 句里找最相似的一对，需要 `C(10000, 2) ≈ 50M` 次推理。而且它压根不产出 Embedding，每次比对都要端到端跑一遍 BERT，没法预先索引。

### SBERT 的动机

Reimers & Gurevych 在 2019 年提出 **Sentence-BERT (SBERT)**，思路非常简单——**把 BERT 放进 Siamese（孪生）架构里，显式用对比学习目标去训句向量本身**。

![SBERT Siamese 架构](/book-notes/hands-on-llm/images/embed-sbert.png)

一句话：**两个共享权重的 BERT + Pooling + 余弦相似度 Loss**。训完之后，单独拿出一侧 BERT 就是一个能输出高质量句向量的 Bi-Encoder。

Bi-Encoder 的性能通常略逊于 Cross-Encoder（因为两个句子从未在注意力里直接交互），但推理快 N 倍——10,000 句只需要 10,000 次编码，再加一次余弦相似度矩阵乘法。

## 第二部分：对比学习——pull together, push apart

### 核心直觉

对比学习的核心思想用两个动词就能概括：

- **Pull together**：把语义相似的句子在向量空间里拉近
- **Push apart**：把语义不同的句子在向量空间里推远

书里用了一个很形象的类比——要让模型理解"马"这个概念，只给正例（四条腿、鬃毛、有尾巴）是不够的，因为这些特征斑马、驴也都有。必须让它**看到反例**（"马不是斑马，因为它没有条纹"），才能学到真正的判别性特征。

![对比学习：用反例教模型什么是马](/book-notes/hands-on-llm/images/embed-contrastive.png)

其实你早就见过对比学习——**Word2Vec 的负采样**就是最早把对比学习用在 NLP 的成功案例：中心词和上下文词是正例，中心词和随机词是负例。

### 三种主流 Loss：一张表看懂

sentence-transformers 里有十几种 Loss，实际用得最多的是三种：

| Loss | 数据形式 | 核心思路 | 适用场景 | 样本效率 |
|------|---------|---------|---------|---------|
| **SoftmaxLoss** | `(sent1, sent2, label)` 三分类 | 把 `[u, v, |u-v|]` 过 Softmax | 早期 SBERT 默认，已不推荐 | 低 |
| **CosineSimilarityLoss** | `(sent1, sent2, score)` score∈[0,1] | MSE 回归预测的余弦相似度到 label | STS-B 这种连续相似度标注 | 中 |
| **TripletLoss** | `(anchor, positive, negative)` | 让 `d(a,p) + margin < d(a,n)` | 有明确正负三元组 | 中 |
| **MultipleNegativesRankingLoss (MNR)** | `(anchor, positive)` 或 `(a, p, hard_neg)` | 批内其他 positive 当 in-batch negative | **最常用**，数据最简单 | 高 |

### MNR Loss：为什么它是默认选择

MNR Loss（也叫 InfoNCE、NT-Xent）的优雅之处：**你只需要提供正样本对**（问题-答案、标题-摘要、图像-说明），负样本从 batch 里其他 pair 自动采样。

![MNR Loss：批内其他答案就是负例](/book-notes/hands-on-llm/images/embed-mnr-loss.png)

一个 batch 有 32 个 `(Q, A)` 对，对每个 Q 来说：
- 正例：它自己的 A
- 负例：batch 里其他 31 个 A（**in-batch negatives**）

优化目标是让 Q 对自己 A 的相似度在这 32 个候选中最高，本质是一个 32 类分类的交叉熵。

::: tip Batch size 越大效果越好
MNR 的任务难度取决于 batch 内的"选项数"。batch=32 是从 32 选 1，batch=128 就是从 128 选 1，难度更大、学习信号更强。这就是为什么训 Embedding 模型的 batch size 通常开到 128、256 甚至上千——只要显存够。
:::

### Hard Negative：in-batch 负例不够用时

in-batch negatives 是"easy negatives"——随机采的，大概率和 anchor 毫不相关，模型很容易区分。想让模型学到更细微的语义差别，要手动构造 **hard negatives**：

- **Easy negative**：`"How many people live in Amsterdam?"` vs `"He was waiting in line for the bus"`——完全无关
- **Semi-hard negative**：`"The capital of the Netherlands is Amsterdam"`——话题相关但不是答案
- **Hard negative**：`"More than a million people live in Utrecht, which is more than in Amsterdam"`——几乎是答案，但说的是另一座城市

Hard negative 通常通过"跑一遍 Cross-Encoder 挑分数高但 label 为负的 pair"或者"让 LLM 生成"来构造。

### 训练代码：用 MNLI 做对比学习数据

NLI（自然语言推理）数据集天然适合做对比学习数据源——`entailment` 是正例，`contradiction` 是负例。

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset, load_dataset
import random
from tqdm import tqdm

# 1. 载入 MNLI，只保留 entailment，构造 (anchor, positive, negative) 三元组
mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
mnli = mnli.filter(lambda x: x["label"] == 0)  # 只要 entailment

train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = list(mnli["hypothesis"])
random.shuffle(soft_negatives)
for row, soft_neg in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_neg)
train_dataset = Dataset.from_dict(train_dataset)

# 2. 选一个 BERT 底座
embedding_model = SentenceTransformer("bert-base-uncased")

# 3. MNR Loss
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

# 4. 训练参数
args = SentenceTransformerTrainingArguments(
    output_dir="mnrloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    warmup_steps=100,
    fp16=True,
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)
trainer.train()
```

书里对比了三种 Loss 在 STS-B 测试集上的 Pearson 相关系数（越高越好）：

| 配置 | Pearson cosine |
|------|---------------|
| Softmax Loss (原始 SBERT) | 0.59 |
| CosineSimilarityLoss | 0.72 |
| **MNR Loss** | **0.80** |

MNR 在同样数据量下显著领先，这也是为什么如今几乎所有新的 Embedding 模型都用 MNR（或其变体）训练。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/contrastive-loss.html" height="600px" title="对比学习三种 Loss 的作用" />

### MTEB：评估 Embedding 模型的统一基准

STS-B 只是一个任务。真实世界里 Embedding 要用于**检索、聚类、分类、重排**等多种任务，单一指标容易误导。

Hugging Face 的 **MTEB（Massive Text Embedding Benchmark）** 横跨 8 类任务、58 个数据集、112 种语言，是目前 Embedding 领域的事实标准 leaderboard。新模型发布几乎都要去刷一下 MTEB。

```python
from mteb import MTEB
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(embedding_model)
```

## 第三部分：数据不够用？TSDAE 与 Augmented SBERT

### Augmented SBERT：用 Cross-Encoder 自动标注

真实场景里往往只有几千条标注，但训一个 Bi-Encoder 需要几十万 pair。**Augmented SBERT** 给出的解法很聪明——**用慢而准的 Cross-Encoder 去给大量无标注 pair 打标，再拿这批"银标"数据去训 Bi-Encoder**。

流程：

1. **Gold dataset**：少量（比如 10k）人工标注的 pair
2. 用 Gold 训一个 Cross-Encoder（BERT + 分类头）
3. 生成大量候选 pair（可以从已有 gold 随机组合，或用预训练 Embedding 做 top-k 召回）
4. 让 Cross-Encoder 给这些候选打标 → **Silver dataset**
5. 用 Gold + Silver 一起训 Bi-Encoder（SBERT）

书里用 10k Gold 训 Bi-Encoder 得分 0.65，配合 40k Silver 做 Augmented SBERT 后得分 0.71——**只用原始 20% 数据量达到近似全量训练的效果**。

### TSDAE：完全无监督的 Embedding 训练

如果连一条标注都没有呢？**TSDAE（Transformer-based Sequential Denoising Auto-Encoder）** 的思路借鉴了 MLM——只是把"遮住单词"换成"删掉单词"：

![TSDAE：加噪 → 编码 → 解码还原](/book-notes/hands-on-llm/images/embed-tsdae.png)

1. 随机从句子里删掉 ~60% 的词，得到"damaged sentence"
2. Encoder（BERT）把 damaged sentence 编码成 **CLS 向量**
3. Decoder 从这一个向量出发，试图还原**原始完整句子**
4. Cross-entropy Loss 作用在 Decoder 输出上

直觉是：**要从 CLS 向量里还原出全句，CLS 必须是一个信息密度极高的语义表示**。训完之后扔掉 Decoder，Encoder 就是你的 Embedding 模型。

::: tip 为什么 TSDAE 用 CLS pooling 而不是 Mean pooling？
MNR 时通常用 Mean pooling 效果更好，但 TSDAE 作者实验发现用 CLS 更好——直觉解释是 Mean pooling 会丢失位置信息，而 Decoder 还原句子恰好需要位置信息，所以 CLS 的表示会被迫保留更多结构信号。
:::

```python
from sentence_transformers import models, SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset

# 用 CLS pooling，注意！
word_emb = models.Transformer("bert-base-uncased")
pooling = models.Pooling(word_emb.get_word_embedding_dimension(), "cls")
embedding_model = SentenceTransformer(modules=[word_emb, pooling])

# 加噪
damaged_data = DenoisingAutoEncoderDataset(list(set(flat_sentences)))

# 去噪 Loss，绑定 encoder 和 decoder 权重
train_loss = losses.DenoisingAutoEncoderLoss(
    embedding_model, tie_encoder_decoder=True
)
```

TSDAE 在完全无监督的情况下也能做到 0.70 的 Pearson 分数——非常接近有监督基线 0.72。

### 领域适配：TSDAE + 监督微调

三种方法的关系：

| 场景 | 推荐方案 |
|------|---------|
| 有大量标注 | 直接监督训练（MNR Loss） |
| 标注少（几千条） | Augmented SBERT |
| 只有领域语料，完全没标注 | **TSDAE 预训练 + 小量监督微调** |

第三种是**领域适配（Domain Adaptation）**的标准流程——先用 TSDAE 在目标领域大规模语料上学领域词和句式，再用少量跨领域标注数据做监督微调。书里管这套叫 **Adaptive Pretraining**。

---

第 11 章把焦点从"造 Embedding"切换到"用表示模型做分类"。

## 第四部分：监督微调 BERT 做分类

### 从 frozen 到 trainable

第 4 章用 `pipeline("sentiment-analysis")` 直接拿预训练模型做分类——**所有层都是 frozen**，只起特征提取作用。第 11 章把 BERT 也放开训练：

![Frozen vs Trainable 架构对比](/book-notes/hands-on-llm/images/embed-freeze-layers.png)

区别只有两点：
- 把整个 BERT 设为 `requires_grad=True`
- 上面接一个 2 类分类头，一起反向传播

在 Rotten Tomatoes（5331 正 + 5331 负影评）上跑一遍，F1 从第 4 章 "frozen 特征 + sklearn 分类器" 的 0.80 提升到 **0.85**。代价只是几分钟 GPU 时间。

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

model_id = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

training_args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
```

### 冻结层技巧：算力不够时的折中方案

BERT-base 有 12 层 encoder，每层约 7M 参数，加起来 110M。全量微调在小数据上容易过拟合、在大数据上又算力紧张。**冻结前 N 层**是一个经典折中：

```python
# 冻结 encoder 的前 10 层，只训最后 2 层 + 分类头
for index, (name, param) in enumerate(model.named_parameters()):
    if index < 165:  # 第 11 个 encoder block 从 index 165 开始
        param.requires_grad = False
```

书里做了个有趣的消融——**逐步放开可训练层数**，看 F1 随之怎么变：

| 可训练 block | F1 |
|-------------|------|
| 全 frozen（只训分类头） | 0.63 |
| 放开最后 1 个 block | ~0.72 |
| 放开最后 2 个（block 10-11） | 0.80 |
| 放开最后 5 个（block 7-11） | 0.84 |
| 全放开 | 0.85 |

**放开最后 5 层已经能达到全量微调 98% 的性能**，但训练速度快 2 倍。这个 "早期层通用、晚期层任务相关" 的规律是所有 Transformer 微调的通用经验：

::: tip 何时冻结多？何时冻结少？
- **数据少（<1k）**：冻结多一些（比如冻前 10 层），防止过拟合
- **数据中等（1k-100k）**：冻前 6-9 层，放开后面几层
- **数据多（>100k）**：全放开，全量微调
- **算力紧**：冻多一些节省 GPU 时间
:::

## 第五部分：SetFit——少样本分类的最佳方案

### 问题：每类 8 条样本能训吗？

真实项目冷启动常见场景：产品经理说"这是用户反馈分类需求，我们人工标了 16 条数据"。16 条数据 BERT 全量微调？肯定过拟合。Zero-shot？准确率堪忧。

**SetFit**（Tunstall et al., 2022）给出了一个结构优雅的解：**把对比学习和分类头分两步训，利用 Embedding 模型做特征增强**。

![SetFit 三步走](/book-notes/hands-on-llm/images/embed-setfit.png)

### SetFit 三步

**Step 1：从小量标注数据构造海量 pair**

有 16 条标注样本（2 类 × 8 条）？对每一条和其他所有样本组 pair：
- 同类 → positive
- 异类 → negative

16 条样本 → `C(16,2) = 120` 对 pair。SetFit 默认每条生成 20 对，最终可以产生 `20 × 16 × 2 = 640` 对训练数据。**32 条原始标注变成 1280 对 contrastive 训练数据**。

**Step 2：用这些 pair 对比学习微调 Sentence-Transformer**

用标准的 SBERT 对比学习流程微调一个预训练 Embedding 模型（比如 `all-mpnet-base-v2`）。这一步让 Embedding 空间**按当前任务的类别组织**——同类样本的向量聚拢，异类推远。

**Step 3：在微调后的 Embedding 上训一个轻量分类头**

原始 16 条样本过新 Embedding 模型，得到 16 个向量和对应 label，训一个 Logistic Regression（默认）或 MLP。

### 代码极简

```python
from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitArgs
from setfit import sample_dataset

# 每类采 16 条
sampled_train = sample_dataset(tomatoes["train"], num_samples=16)

model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

args = SetFitArgs(num_epochs=3, num_iterations=20)
trainer = SetFitTrainer(
    model=model, args=args,
    train_dataset=sampled_train, eval_dataset=test_data,
    metric="f1",
)
trainer.train()
```

### 惊人的样本效率

在 Rotten Tomatoes 上：

| 方法 | 训练样本 | F1 |
|------|---------|-----|
| 第 4 章 Zero-shot embedding + LR | 0 标注 + 8500 LR 训练数据 | 0.80 |
| 第 11 章 BERT 全量微调 | 8500 | 0.85 |
| **SetFit（16/类）** | **32** | **0.85** |

**只用 32 条标注就达到了 8500 条全量微调的效果**——这是对比学习"放大"数据的威力：32 条 → 1280 对 pair。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/sample-efficiency.html" height="500px" title="不同微调方法的样本效率" />

SetFit 还支持**零样本分类**——用类别名（"happy"、"sad"）生成合成样本（"The example is happy"、"The example is sad"），再走上面三步。

## 第六部分：Continued Pretraining with MLM

### 领域适配的另一条路

BERT 在维基百科、图书语料上预训练，对"医疗报告"、"法律合同"、"金融公告"这些领域词汇不熟悉。一个简单但有效的技巧——**在目标领域语料上继续跑 MLM 任务**，然后再微调到具体任务。

这就是 "三步走"：

1. **Pretraining**（别人做好的，比如 `bert-base-cased`）
2. **Continued pretraining** on ACME corp data，用 MLM
3. **Fine-tuning** on classification/NER/retrieval

这条 pipeline 被 BioBERT、LegalBERT、FinBERT 反复验证——**在目标领域语料上 MLM 几个 epoch，后续任务准确率平均能再提 1-3 个点**。

### 代码

```python
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
)

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    "model", learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
```

### Token masking vs Whole-word masking

BERT 的原始 MLM 随机遮 15% 的 token。但 WordPiece 会把 "vocalization" 切成 `vocal` + `##ization`——这时候只遮 `##ization` 让任务变得过于简单（模型看到 `vocal` 就能猜出来）。

**Whole-Word Masking** 遮就遮整个词，难度更高、学到的表示更准。代价是收敛更慢。

## 第七部分：NER——token 级别的分类

### 和句级分类的区别

前面讨论的都是**document-level classification**：输入一整句话，输出一个类别。**NER (Named Entity Recognition)** 不一样——**每个 token 都要一个 label**，标注人名、地名、机构名等。

![NER：每个 token 一个标签](/book-notes/hands-on-llm/images/embed-ner.png)

### BIO 标注方案

CoNLL-2003 数据集用 **BIO** 编码：

```
label2id = {
    "O": 0,           # outside，非实体
    "B-PER": 1,       # 人名的首 token
    "I-PER": 2,       # 人名的后续 token
    "B-ORG": 3, "I-ORG": 4,   # 机构
    "B-LOC": 5, "I-LOC": 6,   # 地点
    "B-MISC": 7, "I-MISC": 8, # 其他
}
```

`"Dean Palmer hit his 30th homer for the Rangers"` 对应：
```
Dean   Palmer  hit  his  30th  homer  for  the  Rangers
B-PER  I-PER   O    O    O     O      O    O    B-LOC
```

B (Begin) 标示短语开始，I (Inside) 标示短语内部。这样 "Dean Palmer" 两个 token 就能合并成一个人名。

### Tricky Point：tokenizer 拆词和标签对齐

实体有个棘手的细节：**BERT 的 tokenizer 会把单词拆成 subword**。`"Maarten"` 会被拆成 `Ma`, `##arte`, `##n` 三个 token。原始标注是词级别的——一个词 `Maarten` 对应一个标签 `B-PER`。但模型按 token 算 loss，需要把词级标签**对齐到 subword 级**。

对齐规则：
- 首 subword 继承原标签（`Ma → B-PER`）
- 后续 subword 用 **I-变体**（`##arte → I-PER`, `##n → I-PER`）
- 特殊 token（`[CLS]`, `[SEP]`）标为 `-100`，cross-entropy 会忽略这类 label

```python
def align_labels(examples):
    token_ids = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = examples["ner_tags"]
    updated_labels = []
    for index, label in enumerate(labels):
        word_ids = token_ids.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx != previous_word_idx:       # 新词的第一个 subword
                previous_word_idx = word_idx
                updated_label = -100 if word_idx is None else label[word_idx]
                label_ids.append(updated_label)
            elif word_idx is None:                  # 特殊 token
                label_ids.append(-100)
            else:                                   # 同一个词的后续 subword
                updated_label = label[word_idx]
                # B-XXX → I-XXX（奇数是 B，+1 变成 I）
                if updated_label % 2 == 1:
                    updated_label += 1
                label_ids.append(updated_label)
        updated_labels.append(label_ids)
    token_ids["labels"] = updated_labels
    return token_ids
```

### Token 分类训练

```python
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
import evaluate

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 用 seqeval 计算 entity-level F1（不是 token-level！）
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)
    # 还原成 string labels，并忽略 -100
    true_preds, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        true_preds.append([id2label[p] for p, l in zip(pred, lab) if l != -100])
        true_labels.append([id2label[l] for _, l in zip(pred, lab) if l != -100])
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {"f1": results["overall_f1"]}
```

::: warning Seqeval vs sklearn
NER 不能用普通 token-level accuracy 评估，因为模型把 "Dean" 标对了但 "Palmer" 标错了，这个**实体整体错了**。`seqeval` 是 entity-level F1——要求整个实体的 BIO 序列都对才算对。
:::

### 推理

训完之后用 `pipeline` 一行搞定：

```python
from transformers import pipeline
token_classifier = pipeline("token-classification", model="ner_model")
token_classifier("My name is Maarten.")
# [{'entity': 'B-PER', 'word': 'Ma',    'start': 11, 'end': 13, 'score': 0.995},
#  {'entity': 'I-PER', 'word': '##arte', 'start': 13, 'end': 17, 'score': 0.993},
#  {'entity': 'I-PER', 'word': '##n',    'start': 17, 'end': 18, 'score': 0.995}]
```

更实用的版本是用 `aggregation_strategy="simple"`，会自动把连续 subword 合并回完整单词和实体。

## 第八部分：把两章串成一张地图

第 10 章和第 11 章讲的其实是同一件事的两面：**BERT/Transformer 作为表示模型（representation model），在自有数据上怎么继续训**。

| 场景 | 技术 | 所属章节 | 数据量 |
|------|------|---------|-------|
| 做一个领域 Embedding 模型 | SBERT + MNR | Ch 10 | 10k-1M pair |
| 领域标注稀缺 | Augmented SBERT | Ch 10 | 几千 gold + 几万 silver |
| 完全无标注 | TSDAE | Ch 10 | 领域纯语料 |
| 监督分类、数据充足 | BERT 全量微调 | Ch 11 | 1k-100k |
| 算力紧或数据少 | BERT 冻结前 N 层 | Ch 11 | 几百到几千 |
| 每类只有 8-32 条 | **SetFit** | Ch 11 | 16-256 条总量 |
| 领域预适配再微调 | Continued MLM pretraining | Ch 11 | 领域语料 |
| Token 级分类 | BERT + seqeval | Ch 11 | 1k-10k 句 |

### 一条实用决策链

我最常用的心智模型是这么一条流程：

1. **有海量标注？** 直接 BERT/RoBERTa 全量微调，完事。
2. **标注中等（几千到几万）？** 先判断是 retrieval 还是 classification——retrieval 走 SBERT + MNR，classification 走 BERT 微调。
3. **标注很少（<100 条）？** 直接上 **SetFit**。
4. **一条标注都没有，但有领域语料？** TSDAE 训 Embedding，或用 LLM 合成标注 + SetFit zero-shot。
5. **预训练底座在你领域很差（医疗、法律）？** 在进入 3/4 之前加一步 **Continued MLM pretraining**。

### 超越本章的视野

这两章讲的都是**表示模型（encoder-only）的微调**，下一章（第 12 章）会转向**生成模型的微调**——SFT、LoRA、QLoRA、RLHF、DPO。两条路线并不矛盾：

- 需要**高质量语义向量做检索/聚类**？Encoder 路线（本章）
- 需要**生成式输出（对话、改写、代码）**？Decoder 路线（下章）
- RAG 系统里两者都要——Encoder 做 retrieval，Decoder 做 generation

真正的生产系统里，一个任务常常会先用 SetFit 做冷启动分类器，上线后积累数据，再升级到 BERT 全量微调，然后为了高并发蒸馏一个更小的 Embedding 模型——**技术选择永远跟着数据量和场景动态演进**。
