---
date: 2026-04-17
title: "Tokens 与 Embeddings：LLM 如何把文本变成数字"
description: "从 BPE 分词算法到 Word2Vec 再到上下文化 Embedding，讲清楚 LLM 表征文本的全链路——以及为什么 tokenizer 的选择会悄悄影响你的模型表现。"
tags: [LLM, Tokenizer, Embedding, BPE, Word2Vec]
---

# Tokens 与 Embeddings：LLM 如何把文本变成数字

> 本文对应原书 **第 2 章 Tokens and Embeddings**，覆盖：分词器算法、Token Embedding、句子/文档 Embedding、Word2Vec 对比训练、Embedding 在推荐系统中的应用。

语言模型其实不"读"文本。它读的是一串 **整数**。

你输入 `"Have the bards who preceded me left any theme unsung?"`，到了 GPU 里实际上变成了 `[1, 14350, 385, 4876, 27746, 5281, 304, 19235, 363, 278, 25305, 293, 16423, ...]` 这样一个张量。这一整套"文本 → 整数 → 向量 → 模型处理 → 整数 → 文本"的管道，就是本章要拆解的内容。

![LLM 输入输出流水线总览](/book-notes/hands-on-llm/images/tokens-pipeline-overview.png)

理解这一层有两个直接收益：

1. **调试能力**：当模型对中文、代码、数字表现奇怪时，多半能在 tokenizer 里找到线索。
2. **成本控制**：`$0.002 / 1K tokens` 的账单，准确估算需要先懂 tokenizer。

---

## 1. Tokenizer 的角色：LLM 的输入输出接口

从外部看，生成式 LLM 接收一段 prompt 返回一段回复；但在内部，prompt 必须先经过 **tokenizer** 切分成 token，每个 token 再被映射为一个整数 ID。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda", torch_dtype="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
# tensor([[ 1, 14350, 385, 4876, 27746, 5281, 304, 19235, 363, 278, ...]])
```

遍历一下 `input_ids`，你会看到它把 `apologizing` 切成了 `apolog / izing`，`tragic` 切成了 `trag / ic`，`gardening` 切成了 `garden / ing`。这就是所谓的 **subword tokenization**——英文常用词整词保留，长词或不常见词拆成多个子词。

> [!tip] 为什么不直接用字符或单词？
> - **单词粒度（Word token）**：词表会爆炸——英文至少几十万个变形；而且遇到新词（OOV）就废。
> - **字符粒度（Character token）**：词表极小（几十到几千），但模型要承担"拼写+语义"两层压力，同样的上下文长度能塞下的信息少得多（经验上字符级比子词级少 **3 倍** 容量）。
> - **子词粒度（Subword token）**：折中方案——常见词整词保留、罕见词拆解、新词拼接——兼顾词表大小与 OOV 鲁棒性，是现代 LLM 的默认选择。

为什么"输入接口"这件小事值得单独开一章？因为 **tokenizer 一旦训练完就和模型绑死了**：你没法把 Phi-3 的 tokenizer 换成 Llama 的，不然 embedding 表的每一行就对不上了。模型的 tokenizer 决定了它对文本的"原始感知"。

---

## 2. 四种 Token 粒度对比

![四种分词粒度对比](/book-notes/hands-on-llm/images/tokens-tokenization-methods.png)

| 粒度 | 典型代表 | 词表大小 | 优势 | 劣势 |
|------|---------|---------|------|------|
| **Word** | Word2Vec、早期 NLP | 数十万到百万 | 每个 token 即一个完整语义单元 | OOV 问题严重；无法处理词形变化（apology/apologize/apologist 各占一格） |
| **Subword** | BPE、WordPiece、Unigram | 3 万~10 万 | 兼顾词表紧凑与 OOV 鲁棒性，现代 LLM 默认 | 生僻词被拆散；中文/日文等无空格语言需额外处理 |
| **Character** | CANINE 等 | 数十~数千 | 完全无 OOV、跨语言通用 | 序列变长，单位上下文能装的信息少；语义压力大 |
| **Byte** | ByT5、GPT-2 byte-level BPE 的 fallback | 256 | 真正的"无 tokenization"；对多语言极友好 | 序列最长，且每个字节本身不携带语义 |

书中给出一个直观的对比例子（Figure 2-6）：同一句 `Have the 🎵 bards who preceded...` 在四种粒度下的切分——字符级和字节级把 emoji 彻底展开，子词级在保持可读性的同时把 `bards` 拆成 `bard + s`。

> [!info] 子词 tokenizer 也会 fallback 到字节
> 现代 BPE tokenizer（如 GPT-2、RoBERTa、GPT-4）虽然是子词级的，但会在**词表中保留 256 个字节作为 fallback**。遇到词表里没有的字符（比如一个冷门 emoji）就用多个字节 token 拼出来。所以它们其实是 "byte-level BPE"，既保留子词效率，又避免了真正的 OOV。

---

## 3. 三种主流分词算法

从"输入 token 粒度"上升一层，还有一个问题：**怎么从训练语料里学出这个子词词表？** 业界有三种主流算法：

| 算法 | 核心思想 | 谁在用 |
|------|---------|--------|
| **BPE**（Byte Pair Encoding） | 从字符/字节开始，**反复合并出现最频繁的相邻 token 对**，直到词表达到目标大小 | GPT-2/3/4、Llama、StarCoder、Galactica |
| **WordPiece** | 类似 BPE，但合并标准不是频次，而是**合并后使训练集似然提升最多的 pair** | BERT、DistilBERT |
| **SentencePiece** | 把空格也视作普通字符一同编码；底层可切换 BPE 或 Unigram LM（基于概率采样子词） | T5、Flan-T5、ALBERT、XLNet、Llama 的部分变体 |

**BPE 的工作原理（直觉版）**：

```
初始词表：{'a', 'b', 'c', ...}
语料：low, lower, newest, widest, ...

Step 1: 统计相邻字符对频次 → ('e','s') 出现最多 → 合并成 "es"
Step 2: 更新后再数 → ('es','t') 最多 → 合并成 "est"
Step 3: 再数 → ('l','o') 最多 → 合并成 "lo"
...
迭代直到词表达到 50K（或你设定的上限）
```

**WordPiece 与 BPE 的差别**：WordPiece 不看"频次最高"，而是看"合并后语言模型 likelihood 提升最大"——它的选择偏向**让整个数据集概率最大化**的 merge。这就是为什么 BERT 的 tokenizer 对词形变化处理得更精细：`##ization`、`##izing` 这种词尾专门被抽成一个子词。

**SentencePiece 的工程价值**：它把"空格"也看作普通字符，因此对中文、日文、泰文这些无空格语言天然友好——不需要先分词再子词化。Llama、T5 选它也是出于多语言考虑。

> [!warning] Unigram LM 是另一条路
> SentencePiece 还支持 Unigram 语言模型分词（Kudo 2018）：它先建一个"超大"候选子词集合，然后迭代删除"对 likelihood 贡献最小"的子词，直到词表缩到目标大小。它能在推理时对同一句话产生多种分词，常用于 **subword regularization**——训练时随机选一种切法，作为数据增强。

---

## 4. 实战：同一句话在不同 LLM 里长什么样

书中做了一个教科书级的对比实验：用同一段混杂了英文大小写、中文、emoji、Python 代码、数字的文本，喂给从 2018 年到 2024 年的各代 LLM tokenizer，观察差异。

```python
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""
```

![不同 tokenizer 的切分对比](/book-notes/hands-on-llm/images/tokens-tokenizer-compare.png)

把书中结论提炼成对比表：

| Tokenizer | 年份 | 算法 | 词表大小 | 对 CAPITALIZATION | 对中文/emoji | 对空格 | 对数字 |
|-----------|------|------|---------|-------------------|-------------|-------|--------|
| **BERT-uncased** | 2018 | WordPiece | 30,522 | 全部转小写，`capital + ##ization` | 中文/emoji 全部 `[UNK]` | 换行丢失，空格无 token | 按 WordPiece 切 |
| **BERT-cased** | 2018 | WordPiece | 28,996 | 保留大小写但切 8 个 subword | 同上 | 换行丢失 | 同上 |
| **GPT-2** | 2019 | byte-level BPE | 50,257 | 保留，`CAP / ITAL / IZ / ATION` | 中文/emoji 按字节拆成多个 token | 每个连续空格一个 token | 数字可能被合并（如 870 一个 token，871 拆两个） |
| **Flan-T5** | 2022 | SentencePiece (BPE) | 32,100 | `CA / PI / TAL / IZ / ATION` | 中文/emoji 变 `<unk>`（完全失明）| 无换行、无空格 token | - |
| **GPT-4** | 2023 | BPE | ~100,000 | `CAPITAL / IZATION`（2 个） | emoji 拆字节 | **专门的多空格 token**（最多 83 空格） | 类似 GPT-2 |
| **StarCoder2** | 2024 | BPE | 49,152 | `CAPITAL / IZATION` | emoji 拆字节 | 类似 GPT-4 | **每位数字独占一个 token**（600 → 6,0,0） |
| **Phi-3 / Llama 2** | 2023/24 | BPE | 32,000 | 类似 GPT-4 | emoji 拆字节 | 保留空格 | - |

几个核心观察：

**① 大小写处理**。BERT-uncased 直接把所有字母转小写，意味着 "Apple" 和 "apple" 模型眼里一模一样——这对 NER（人名、公司名）是致命的。现代 LLM 都选择保留大小写。

**② 代码友好度**。GPT-4 专门为 `elif` 这种 Python 关键字建了单独的 token，而且对多重缩进空格有专门处理。StarCoder2 更激进——它让每一位数字独占 token，因为作者相信这样能提升模型对数字和数学的表示能力。

**③ 对多语言和 emoji 的敏感度**。Flan-T5 把所有中文和 emoji 都变成 `<unk>`——这意味着它对这些内容完全失明，不可能生成或理解。GPT-2 之后的所有 byte-level BPE tokenizer 都能通过字节拼出这些字符，只是 "效率"不同——中文字符可能占 2~3 个 token。

**④ 词表在持续变大**。BERT 时代 30K，GPT-2 时代 50K，GPT-4 已经 100K+。更大的词表意味着单个 token 能携带更多信息、相同文本占用更少 token、推理更便宜——但代价是 embedding 矩阵更大。

> [!tip] 选 tokenizer 的工程建议
> - **多语言场景**：选词表大、byte-level fallback 支持好的（GPT-4 / Llama / Qwen 系）；避开 Flan-T5 这种把非训练语言扫成 `<unk>` 的。
> - **代码场景**：优先选 StarCoder2、Code Llama、DeepSeek-Coder 等——它们对缩进、数字、关键字做了专门优化。
> - **成本敏感场景**：用 `tiktoken` / `tokenizers` 先跑一遍你的典型输入，算出实际 token 数 × 单价，再决定选哪个模型。同样一段中文输入，Claude 的 token 数可能比 GPT-4 多 30%。

---

## 5. Token Embedding：查表矩阵的本质

Tokenizer 把文本变成整数 ID 之后，语言模型要做的第一件事就是把这些 ID 查成向量——这张表就是 **embedding 矩阵**。

![LLM 内部的 embedding 矩阵](/book-notes/hands-on-llm/images/tokens-embedding-matrix.png)

本质上它是一个形状为 `[vocab_size, embedding_dim]` 的参数矩阵。每一个 token ID 对应矩阵的一行：

```python
# 伪代码：embedding lookup 的本质
embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))

def embed(token_ids):  # [batch, seq_len]
    return embedding_matrix[token_ids]  # [batch, seq_len, embedding_dim]
```

这几个数值对参数量估算很关键：

| 模型 | 词表 | hidden_dim | Embedding 参数量 |
|------|------|-----------|------------------|
| BERT-base | 30,522 | 768 | ~23.4 M |
| GPT-2 small | 50,257 | 768 | ~38.6 M |
| Llama-2 7B | 32,000 | 4,096 | ~131 M |
| Llama-3 8B | 128,256 | 4,096 | ~525 M |
| GPT-4（推测） | ~100,000 | ~12,288 | ~1.2 B |

Llama-3 从 32K 扩展到 128K 词表，embedding 表直接从 131M 涨到 525M——词表不是免费午餐。训练开始前这些参数是**随机初始化**的，和模型其他权重一样；在预训练过程中才被训练到携带语言信息。

> [!info] Tie-embedding 技巧
> 很多模型让输入 embedding 表和输出 LM head（预测下一个 token 的分类头）**共享同一组参数**——这是因为二者本质上都是 `vocab_size × hidden_dim` 的矩阵。绑定后参数量减半，还能让"词的输入表示"和"词的输出概率"保持一致性。GPT-2、T5 都用了；Llama 系列出于性能考虑没绑。

---

## 6. 上下文化 Embedding：为什么 "bank" 不只有一个向量

静态 embedding（Word2Vec、GloVe）有一个根本问题：**同一个词只有一个向量**。但在真实语言里，`"river bank"` 和 `"money bank"` 里的 `bank` 完全是两个意思。

![Transformer 产生上下文化 embedding](/book-notes/hands-on-llm/images/tokens-contextual-embeddings.png)

**语言模型创造的 token embedding 是"上下文化"的**——同一个 token 在不同上下文里会被映射到不同的向量。这是通过 Attention 实现的：每一层 Transformer 都在"根据上下文更新每个 token 的表示"。

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

tokens = tokenizer("Hello world", return_tensors="pt")
output = model(**tokens)[0]
print(output.shape)  # torch.Size([1, 4, 384])
# 4 个 token（[CLS], Hello, world, [SEP]），每个 384 维
```

`output` 是每一层经过 attention 后的 **上下文化表示**——不是静态查表结果，而是"考虑了整句话"之后的动态向量。NER（命名实体识别）、抽取式摘要、情感分析这些任务，都是在这个 contextual embedding 之上搭的分类头或指针网络。

> [!tip] 上下文化是 BERT 家族的灵魂
> ELMo（2018）第一次把"contextual"概念带进主流；BERT（2018）用双向 Transformer 把它做到了工业可用；DeBERTa、RoBERTa、DistilBERT 在这条路上继续优化。现在我们用的每一个"代表性"模型（用来做分类、检索、聚类）——本质都是 contextual embedding 生成器。

---

## 7. Text Embedding：一句话或一篇文档的向量

Token embedding 是一串向量（每个 token 一个）。但很多下游任务——语义搜索、RAG、文本聚类——只需要 **一段文本整体一个向量**。这就是 **text embedding** 的职责。

怎么从"一串向量"得到"一个向量"？常见的 pooling 策略有三种：

| 策略 | 做法 | 典型模型 |
|------|------|---------|
| **[CLS] pooling** | 取 `[CLS]` 特殊 token 对应的最后一层输出 | 原版 BERT 分类任务 |
| **Mean pooling** | 所有 token 的最后层向量取平均 | Sentence-BERT、大多数现代 embedding 模型 |
| **Weighted pooling** | 按 attention 权重加权平均，或仅对非 pad token 平均 | 各种 SOTA 句向量模型的细节改进 |

高质量的 text embedding 模型往往**为这个任务专门训练**——不能直接拿 BERT 的 mean pooling 当 text embedding（效果会很差，参见 *Sentence-BERT* 论文的 STS benchmark 对比）。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
vector = model.encode("Best movie ever!")
print(vector.shape)  # (768,)
```

这类模型的训练方法本质上是 **对比学习**：让相似句子在向量空间里靠近、不相似的远离。第 10 章会深入讲，本章只是埋线。

---

## 8. Word2Vec 与对比学习：Embedding 训练的"祖师爷"

在 Transformer 一统江湖之前，**Word2Vec**（Mikolov 2013）是主流的静态 word embedding 方法。理解它不是为了"考古"，而是因为它的训练范式——**对比学习 + 负采样**——至今是 embedding 模型训练的基本功。

![Word2Vec skip-gram 与负采样](/book-notes/hands-on-llm/images/tokens-word2vec-skipgram.png)

**Skip-gram 的思路**：给定一个中心词，预测它周围的词。用滑动窗口从语料里生成训练样本：

```
文本："Thou shalt not make a machine in the likeness..."
窗口大小 = 2

中心词 not 生成 4 条正例：(not, thou, 1), (not, shalt, 1), (not, make, 1), (not, a, 1)
中心词 make 生成 4 条正例：(make, shalt, 1), (make, not, 1), (make, a, 1), (make, machine, 1)
...
```

但全都是正例，模型会"学会"永远输出 1。所以要加 **负采样**（Negative Sampling）：

```
(not, thou, 1)          ← 正例
(not, apothecary, 0)    ← 随机配一个不相关的词
(not, sublime, 0)       ← 再配一个
(make, def, 0)          ← 再配一个
```

一个二分类模型在这样的数据上训练：输入两个词，输出它们是否是"邻居"的概率。**训练过程不更新最后的分类头，而是不断调整输入的两个词 embedding**——相关词越来越近，无关词越来越远。训练结束后，embedding 矩阵本身就是我们想要的产物。

> [!info] Noise Contrastive Estimation (NCE) 的思想
> Word2Vec 的负采样是一个叫 **NCE（噪声对比估计）** 的框架的特例：与其算一个昂贵的归一化概率（分母要遍历整个词表），不如训练一个二分类器区分"真数据"和"噪声采样"。这套思路后来被用到 SBERT、SimCLR、CLIP、DPO 等一大堆方法里——**对比学习是当代表征学习的默认范式**。

---

## 9. Embedding 的非 NLP 应用：歌曲推荐

Word2Vec 的思想可以推广到任何"序列数据"。**把每首歌当作一个 token，把用户播放列表当作一个句子**——训出来的 embedding 空间里，相似歌曲自然靠得近。这就是 Spotify 等推荐系统的经典做法。

```python
import pandas as pd
from urllib import request
from gensim.models import Word2Vec

# 下载包含大量播放列表的公开数据集（Shuo Chen @ Cornell）
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
lines = data.read().decode("utf-8").split('\n')[2:]
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# 训练 Word2Vec：把每个 playlist 当作一个"句子"
model = Word2Vec(playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)
```

喂给它 Metallica 的 `Fade To Black`（ID 2172），它会推荐：

```
Little Guitars     — Van Halen
Unchained          — Van Halen
The Last in Line   — Dio
Mr. Brownstone     — Guns N' Roses
Breaking the Law   — Judas Priest
```

全是经典重金属/硬摇滚。模型完全没看过歌词、没看过 genre 标签——它只是根据"哪些歌常出现在同一个 playlist 里"学到了歌曲间的语义相似性。

> [!tip] 这就是现代推荐系统的一层 baseline
> 工业界的 Item2Vec、Prod2Vec（Amazon）、Airbnb Listings Embedding、YouTube Video Embedding，本质上都是这一思路的工程化：**把用户行为序列当文本，把 item 当 token，用 Word2Vec / Transformer 训 item embedding**。再把这些 embedding 喂给 ANN 索引（FAISS、ScaNN、HNSW）做在线检索。

---

## 10. 工程清单：挑 tokenizer / embedding 时该想什么

整理一个实战 checklist：

### 选 tokenizer 时问自己三个问题

1. **语言分布**：我的输入里中文/日文/非英语占比多少？如果 > 20%，优先选 GPT-4 / Llama 3 / Qwen 系列（大词表 + byte fallback）。
2. **是否有代码**：有代码的话 StarCoder / Code Llama / DeepSeek-Coder 会显著省 token（每位数字和多空格都有专门 token）。
3. **成本模型**：用 `tiktoken.encoding_for_model(...)` 或 `AutoTokenizer` 跑一遍你的典型输入，把 token 数 × 单价算清楚——再决定。

### 选 text embedding 模型时问自己三个问题

1. **MTEB 排行榜上的表现**：[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 是目前最权威的 embedding 模型评测，分 retrieval、classification、clustering 等多个维度。
2. **向量维度与成本**：维度越高检索精度越好但存储和计算贵。主流范围 384（轻量）、768（标准）、1024~4096（高质量）。
3. **领域适配**：通用模型 vs. 法律/医学/代码专用模型——垂直领域的 domain-specific embedding（如 BGE-code、Voyage-law）往往能带来 5-15 个点的提升。

---

## 小结

| 主题 | 核心要点 |
|------|---------|
| Tokenizer | 把文本变整数的接口；和模型绑死，训练后不可换 |
| 粒度选择 | 现代 LLM 几乎全用 subword；byte-level fallback 处理未知字符 |
| 主流算法 | BPE（最常用）、WordPiece（BERT）、SentencePiece（多语言/代码） |
| Tokenizer 演进 | 词表从 30K → 100K+；保留大小写、支持多空格、对数字做专门处理 |
| Token embedding | 词表 × hidden_dim 的查表矩阵；Llama-3 的 embedding 表已达 525M 参数 |
| Contextual embedding | Transformer 让同一个 token 在不同上下文中有不同向量；`bank` 有多重含义 |
| Text embedding | 用 pooling（CLS / mean / weighted）把 token 序列压成单向量；专门训练效果最佳 |
| Word2Vec | Skip-gram + 负采样；**对比学习是 embedding 训练的默认范式** |
| 非 NLP 应用 | 任何"序列数据"都能用：歌曲、商品、视频、用户行为 |

下一章我们会深入 **Transformer 架构内部**——这个把 token ID 变成 contextual embedding、再变成下一个 token 预测的核心引擎。

---

## 动手玩：Tokenizer Playground

下面这个可视化模拟了 BPE、WordPiece、Byte-level 三种分词方式。输入你自己的文本，或点预设按钮试试英文/中文/代码/emoji 各自会被怎么切。你会发现同样的输入，不同算法的 token 数能差出好几倍。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/tokenizer-playground.html" height="600px" title="Tokenizer 对比实验" />
