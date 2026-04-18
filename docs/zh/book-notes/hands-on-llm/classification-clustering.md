---
date: 2026-04-17
title: "文本分类与聚类：从 BERT 到 BERTopic 的三条路线"
description: "同样是给文本打标签，为什么有时候选 BERT 微调，有时候选 zero-shot embedding，有时候又要交给生成模型？再加上聚类 + 主题建模的 pipeline，把文本理解任务的工具箱一次讲清。"
tags: [文本分类, 文本聚类, BERTopic, BERT, Embedding, LLM 应用]
---

# 文本分类与聚类：从 BERT 到 BERTopic 的三条路线

> 本文对应原书 **第 4 章 Text Classification** 和 **第 5 章 Text Clustering and Topic Modeling**，覆盖：三种分类路线（Representation / Embedding / Generative）、聚类 pipeline（UMAP + HDBSCAN）、BERTopic 模块化主题建模框架。

书里这两章放在一起读很有意思——它们其实在回答同一个问题："给定一堆文本，怎么把它们组织成结构化的信息？" 区别只是：**有标签就叫分类，没标签就叫聚类，希望输出可读标签就叫主题建模**。

而在 LLM 时代，这个老派 NLP 问题的工具箱被彻底重塑：从"训一个专用 BERT"到"套一个 embedding 加余弦相似度"再到"写个 prompt 让 ChatGPT 直接给答案"，三种路线各有各的成本结构，选错了可能差几十倍算力。

本章的价值就在于把这三条路线并排展开，让你能够**按场景选型**，而不是"不管什么任务都上 GPT-4"。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/text-task-router.html" height="620px" title="文本任务决策路由" />

---

## 第一部分：分类的三条路线

### 问题定义

给定文本，输出标签。书里用的是经典的 Rotten Tomatoes 电影评论数据集（正面/负面两分类，5331 + 5331 条），算是"最小可复现"的场景。

```python
from datasets import load_dataset

data = load_dataset("rotten_tomatoes")
# train: 8530 / validation: 1066 / test: 1066
# 每条是 {'text': '...', 'label': 0 或 1}
```

在 LLM 出现之前，标准答案是：TF-IDF + 逻辑回归。书里一上来就提醒：**这个 baseline 依然值得跑一跑**——它没有 GPU 依赖、训练时间秒级、解释性强。后面三条路线都应该跟它对比一下，否则可能在用大炮打蚊子。

接下来，书里把语言模型能做分类的方式分成两大类：**表示模型（representation model）** 和 **生成模型（generative model）**。前者直接输出类别索引，后者生成一段文字（你再解析）。

![分类：表示模型 vs 生成模型](/book-notes/hands-on-llm/images/classify-two-approaches.png)

图里这个对比关系后面每一节都会用到——你可以把它当成"地图"。

---

### 路线 1：Representation Model（任务特定模型）

**最直接的方式**：去 Hugging Face Hub 上找一个**已经为目标任务微调过**的 BERT 类模型，直接 inference。

书里用的是 `cardiffnlp/twitter-roberta-base-sentiment-latest`——虽然它原本是为推特训练的情感分类模型，但在电影评论上也能跑。核心代码只要三行：

```python
from transformers import pipeline

pipe = pipeline(
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True,
    device="cuda:0",
)

# 推理：直接拿到 {negative: 0.02, neutral: 0.08, positive: 0.90} 这样的分数
y_pred = []
for output in pipe(KeyDataset(data["test"], "text")):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    y_pred.append(np.argmax([negative_score, positive_score]))
```

**F1 = 0.80**——对一个"跨领域迁移"的模型来说很不错了。如果换成 `distilbert-base-uncased-finetuned-sst-2-english`（SST-2 本身就是电影评论数据集），分数还能再涨。

::: tip 什么时候选这条路
- Hub 上**存在**针对你任务已经微调好的现成模型；
- 单机有 GPU 可用；
- 不希望自己训练模型。
:::

**缺点**：如果你的任务是"识别公司内部工单的紧急程度"这类**垂直任务**，基本不可能在 Hub 找到现成模型——这时路线 1 就不可行。

---

### 路线 2：Embedding + 轻量分类器

换个思路：**embedding 模型**（比如 `sentence-transformers/all-mpnet-base-v2`）本来是通用工具，只负责把文本变成 768 维向量。那我们能不能把这些向量当成特征，**在上面训一个轻量分类器**？

这就是"把特征抽取和分类器解耦"——书里称为两步法（two-step approach）：

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Step 1: 用 embedding 模型把文本变成向量（不训练，GPU 或 API 都行）
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
# train_embeddings.shape == (8530, 768)

# Step 2: 只在 CPU 上训一个逻辑回归
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])
y_pred = clf.predict(test_embeddings)
```

**F1 = 0.85**——比路线 1 的跨域 RoBERTa 还高一点！而且第二步可以完全在 CPU 上跑，成本极低。如果把 embedding 也换成 Cohere/OpenAI 的 API，整条 pipeline 甚至不需要自己持有 GPU。

更有意思的是 **Zero-Shot 变体**：连逻辑回归都不用训。思路是——把**标签本身**也 embedding 化：

![Zero-shot 分类：把标签 embedding 化](/book-notes/hands-on-llm/images/classify-zero-shot.png)

把 `"A negative review"` 和 `"A positive review"` 两句话喂给 embedding 模型，得到两个标签向量；然后对每个测试文档，看它跟哪个标签向量的 **余弦相似度** 更高：

```python
from sklearn.metrics.pairwise import cosine_similarity

label_embeddings = model.encode(["A negative review", "A positive review"])
# test_embeddings.shape == (1066, 768)
# label_embeddings.shape == (2, 768)

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)
```

**F1 = 0.78**——在**零标注数据**的情况下达到这个水平已经相当惊人。而且，把 prompt 从 `"A negative review"` 改成 `"A very negative movie review"` 往往能再涨几个点——这已经开始有点 prompt engineering 的味道了。

::: tip 什么时候选这条路
- 没有现成的任务专用模型；
- 只有很少的标注数据（supervised 变体）或完全没有标注（zero-shot 变体）；
- 成本敏感：embedding 可批量计算、可缓存、可用 API。
:::

---

### 路线 3：Generative Model（文本到文本 / 对话模型）

最后一条路是把分类任务转成**文本生成**。书里给了两个例子：

**(1) Flan-T5（开源 encoder-decoder）**

T5 的设计哲学就是"一切任务都是 text-in → text-out"。它的预训练就是混合了上千种任务的 instruction tuning，自带理解自然语言指令的能力。

![T5 架构：encoder-decoder 的 12 + 12 层](/book-notes/hands-on-llm/images/classify-t5.png)

使用方式是给每条文本拼上 prompt：

```python
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-small", device="cuda:0")

prompt = "Is the following sentence positive or negative? "
data = data.map(lambda x: {"t5": prompt + x["text"]})

y_pred = []
for output in pipe(KeyDataset(data["test"], "t5")):
    text = output[0]["generated_text"]  # 直接返回 "positive" 或 "negative"
    y_pred.append(0 if text == "negative" else 1)
```

**F1 = 0.84**（flan-t5-small 这种最小号的版本）。

**(2) ChatGPT（闭源 decoder-only API）**

```python
import openai
client = openai.OpenAI(api_key="YOUR_KEY")

def chatgpt_generation(prompt, document):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.replace("[DOCUMENT]", document)},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content

prompt = """Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""
```

**F1 = 0.91**——断层领先。但成本是：每条样本一次 API 调用，且数据离开了本地。

::: warning 生成模型分类的坑
输出是**字符串**而非 class index，需要解析。`temperature=0` 保证确定性，但模型偶尔会"画蛇添足"输出 `"The answer is: 1"`——prompt 里最好显式要求"Do not give any other answers"。
:::

---

### 三条路线对比

| 路线 | 代表工具 | 标注需求 | 算力成本 | 准确率 | 何时选 |
|------|---------|---------|---------|--------|--------|
| **1. Task-Specific Rep.** | 微调好的 BERT | 否（已完成） | GPU inference | 中-高 | Hub 上已有现成模型 |
| **2a. Embedding + 分类器** | SBERT + LR | 少量即可 | CPU 训练 + GPU/API embedding | 高 | 领域特殊、标注有限 |
| **2b. Embedding Zero-Shot** | SBERT + 余弦 | **完全不要** | 纯 inference | 中 | 原型验证、冷启动 |
| **3a. Generative Open** | Flan-T5 | 否 | GPU inference | 中-高 | 想本地化、任务多变 |
| **3b. Generative API** | GPT-3.5/4 | 否 | 按 token 付费 | 极高 | 预算够、数据脱敏 |

一个朴素的选型直觉：

- **通用任务 + 有现成模型** → 路线 1
- **垂直任务 + 少标注** → 路线 2a
- **原型验证 / 完全冷启动** → 路线 2b
- **任务多变 + 想本地化** → 路线 3a
- **追求极致效果 + 预算充足** → 路线 3b

---

## 第二部分：聚类与主题建模

上一部分的前提都是"你知道有哪些类别"。但真实场景下，很多数据是**先有一大堆文本，后想知道里面有什么主题**——比如你爬了 44949 篇 ArXiv 论文的 abstract，想看看这几年 NLP 领域都在研究什么。

这就是 **聚类（clustering）** 和 **主题建模（topic modeling）** 要解决的问题。两者的区别是：
- **聚类**：给每个文档分配一个 cluster id，同簇内语义相近；
- **主题建模**：在聚类之上，再为每个簇生成**可读的标签**（一串关键词或一句自然语言标题）。

### 聚类 Pipeline 的三段式

书里给的通用 pipeline 只有三步，每步都有清晰的工具选择：

```
文档集合
   │  Step 1: Embedding（Sentence-Transformers 等）
   ▼
[N × 384] 高维向量
   │  Step 2: 降维（UMAP）
   ▼
[N × 5] 低维向量
   │  Step 3: 聚类（HDBSCAN）
   ▼
每个文档 → cluster id（可能是 -1，表示 outlier）
```

**Step 1：Embedding**

跟前面分类任务用的是同一套工具，但模型换成专门优化过聚类任务的 `thenlper/gte-small`（小且快，MTEB clustering 分数高）：

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
# embeddings.shape == (44949, 384)
```

**Step 2：为什么要降维？（UMAP > PCA）**

直接在 384 维上聚类会遇到**维度灾难**——高维空间里，几乎所有点对的距离都差不多，聚类算法找不到有意义的密度区域。

那为什么是 UMAP，不是老牌的 PCA？

| 方法 | 是否保留全局结构 | 是否保留局部结构 | 非线性关系 |
|------|----------------|----------------|-----------|
| PCA | ✅ | ❌ | ❌ |
| t-SNE | ❌ | ✅ | ✅ |
| **UMAP** | ✅ | ✅ | ✅ |

UMAP 的流形假设对"语义相近的文档应该聚成局部邻域"这个诉求是天然契合的。

```python
from umap import UMAP

umap_model = UMAP(
    n_components=5,       # 降到 5 维（不是 2，因为 2 维损失太多）
    min_dist=0.0,         # 让同类点贴得更紧
    metric="cosine",      # 文本语义默认用 cosine
    random_state=42,
)
reduced_embeddings = umap_model.fit_transform(embeddings)
# reduced_embeddings.shape == (44949, 5)
```

::: tip 降维到 5 还是 2？
书里给的经验值是聚类用 **5–10 维**（保留更多结构），可视化用 **2 维**（只为画图）。两次 UMAP 调用互不干扰。
:::

**Step 3：为什么是 HDBSCAN，不是 K-Means？**

K-Means 的两个大问题：
1. 必须预先指定 k（44949 篇论文分几个主题？没人知道）；
2. 强制每个点都属于某个簇——**噪声点会污染簇中心**。

HDBSCAN 的设计恰好解决这两个问题：
- 基于密度：高密度区域自动形成簇，自动决定簇的数量；
- 能识别 **outlier**（label = -1），不强行分配。

![密度聚类 vs 中心点聚类](/book-notes/hands-on-llm/images/cluster-density-vs-centroid.png)

```python
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(
    min_cluster_size=50,          # 少于 50 篇的话题不作为独立簇
    metric="euclidean",           # UMAP 输出后用欧氏距离即可
    cluster_selection_method="eom",
).fit(reduced_embeddings)

clusters = hdbscan_model.labels_
len(set(clusters))  # 156 个簇 + 一个 "-1" outlier 簇
```

人工抽检 cluster 0 的几篇文档 abstract，发现都在讨论手语翻译——算法**自发**发现了一个主题，完全不需要人工定义。

---

### BERTopic：把上面的 pipeline 做成"乐高"

到这里，我们有了聚类结果，但只有一个 cluster id（比如 0、1、2……）——人看不懂。要变成"Topic 0 = automatic speech recognition"这样的可读标签，还缺一步。

这就是 **BERTopic** 要做的事。它的作者就是这本书的作者之一 Maarten Grootendorst。BERTopic 的架构**全是"乐高块"**——每一步都可以替换：

![BERTopic 全流程：5 个乐高块 + 可选的第 6 块](/book-notes/hands-on-llm/images/classify-bertopic.png)

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/bertopic-pipeline.html" height="560px" title="BERTopic 五步管线" />

框架分成两大部分：

**左半 Clustering（创建主题）**：就是前面讲的三段式（SBERT → UMAP → HDBSCAN）。

**右半 Topic Representation（命名主题）**：
1. **CountVectorizer**（bag-of-words）：统计每个簇里每个词出现多少次；
2. **c-TF-IDF**（class-based TF-IDF）：**把整个簇当作一篇"大文档"**，算 TF-IDF。跨簇常见的词（"the", "of"）权重被压低，簇内独有的词（"translation", "speech"）权重被拔高。

这个 c-TF-IDF 是 BERTopic 的核心创新——它是**无监督**的，不需要训练，速度极快。

跑起来异常简单：

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
).fit(abstracts, embeddings)

topic_model.get_topic_info()
# Topic -1: [the, of, and, to, in, we, that, ...]        <- outlier 主题（可忽略）
# Topic  0: [speech, asr, recognition, end, acoustic...] <- 语音识别
# Topic  1: [medical, clinical, biomedical, patient...]  <- 医疗 NLP
# Topic  2: [sentiment, aspect, analysis, reviews...]    <- 情感分析
# Topic  3: [translation, nmt, machine, neural, bleu...] <- 机器翻译
# ...
```

**搜索特定主题**：

```python
topic_model.find_topics("topic modeling")
# ([22, -1, 1, 47, 32], [0.954, 0.911, 0.907, 0.906, 0.905])

topic_model.get_topic(22)
# [('topic', 0.066), ('topics', 0.036), ('lda', 0.016), ('latent', 0.013), ...]
```

---

### 第 6 块乐高：用 LLM 给主题起"人话"标题

c-TF-IDF 输出的依然是一串关键词，不是一个 topic name。BERTopic 的另一层模块化是 **Representation block**——在关键词表之上，再套一个生成模型去"润色"。

比如 **KeyBERTInspired**：对每个簇，把代表文档取平均 embedding，再跟候选关键词的 embedding 算余弦相似度，重新排序：

```python
from bertopic.representation import KeyBERTInspired

topic_model.update_topics(abstracts, representation_model=KeyBERTInspired())
```

如果要"人话标题"（比如 "Text Summarization"），可以换成 **用 LLM 的 Representation block**：

```python
from bertopic.representation import OpenAI

prompt = """I have topic that contains documents: [DOCUMENTS]
The topic is described by the keywords: [KEYWORDS]
Based on the above, give a short label for the topic."""

representation_model = OpenAI(model="gpt-3.5-turbo", prompt=prompt)
topic_model.update_topics(abstracts, representation_model=representation_model)
```

::: tip 为什么不一开始就用 LLM？
因为你有**几万**篇文档，但只有**几百**个主题。如果直接把 LLM 放在 embedding 环节——几万次 API 调用，贵且慢。而 Representation block 是**一个主题只调用一次**，顶多几百次——成本差了两个数量级。

这就是 BERTopic "乐高块"设计的精髓：**把贵的操作放到数据量最少的那层去**。
:::

---

## 小结：一张表看完整本章

| 任务场景 | 推荐工具栈 | 典型代码 |
|---------|----------|---------|
| 有 Hub 现成模型的分类 | `transformers.pipeline` | 3 行 |
| 垂直领域 + 少标签分类 | SBERT + sklearn LR | ~10 行 |
| 无标签的分类原型 | SBERT + cosine similarity | ~5 行 |
| 多任务 / 任务多变 | Flan-T5 / ChatGPT API | prompt 工程为主 |
| 探索未知数据集的主题 | **BERTopic**（SBERT + UMAP + HDBSCAN + c-TF-IDF） | ~8 行 |
| 给主题起 "人话" 标题 | BERTopic + OpenAI Representation block | +3 行 |

两章合起来传达的**最重要的一个判断**是：

> **在 2024 年做文本任务，embedding 模型是整个工具箱里最被低估的组件。**

你可以用它做分类（supervised / zero-shot）、做聚类、做主题建模、做语义搜索（下一章的主题）——而它的成本结构又允许你批量缓存、用 API、在 CPU 上跑。这是一条比"无脑用 GPT"性价比高得多的默认路径。

而 BERTopic 是书里两位作者展示"模块化架构胜利"的一个最典型的案例——不是因为它发明了什么新算法，而是因为它把**已有的五个工具**（SBERT / UMAP / HDBSCAN / CountVectorizer / TF-IDF）用"乐高块"的方式组合起来，允许你**按场景替换任一块**。当 MTEB 上出现更好的 embedding 模型时，你只要换一行代码，整个 pipeline 就升级了。

这种设计理念，某种意义上也是整本 Hands-On LLM 想传达的——**LLM 应用工程的核心技能，不是训练，而是组合。**
