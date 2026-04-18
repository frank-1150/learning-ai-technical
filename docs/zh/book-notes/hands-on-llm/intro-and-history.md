---
date: 2026-04-17
title: "LLM 基础：从 Bag-of-Words 到 GPT 的语言 AI 简史"
description: "沿着 50 年 NLP 演进史，梳理大语言模型为何是当下这个样子——从词袋模型、Word2Vec、注意力机制，到 Encoder-only、Decoder-only 两条架构分支的分野。"
tags: [LLM, NLP, Transformer, 语言模型, 深度学习]
---

# LLM 基础：从 Bag-of-Words 到 GPT 的语言 AI 简史

> 本文对应原书 **第 1 章 An Introduction to Large Language Models**，覆盖：语言 AI 发展简史、从 BoW 到 Transformer 的表征演进、Encoder/Decoder 的分野、LLM 的训练范式。

---

## 1. 什么是 Language AI？

在具体讨论 LLM 之前，先厘清几个常被混用的概念。

**人工智能（AI）** 泛指"让软件表现出接近人类智能的能力"的一类技术。John McCarthy 给过一个经典定义：

> AI 是制造智能机器——尤其是智能计算机程序——的科学与工程。它与用计算机理解人类智能这个任务相关，但 AI 不必局限于那些生物学上可观察的方法。
> — John McCarthy, 2007

**Language AI** 是 AI 的一个子领域，专注于"理解、处理、生成人类语言"。在机器学习方法主导 NLP 的今天，它经常与 **自然语言处理（NLP）** 互换使用。本书作者特意选择 "Language AI" 一词，因为它能涵盖一些技术上不是 LLM、但对整个领域有重要影响的技术（比如检索系统）。

**LLM（大语言模型）** 又是 Language AI 中的一类——大体上指参数量足够大、能够理解或生成语言的神经网络模型。但正如原书指出的，"大"本身是个移动靶：今天的 GPT-3（175B）当然是大，但如果明天出现一个只有 10B 参数、能力却接近 GPT-3 的模型，它还算不算"大"？本书对 LLM 采用较宽松的定义——**既包括生成式 decoder-only 模型，也包括 encoder-only 的表征模型**，只要是基于神经网络并服务于语言 AI 任务的都算。

> [!note]
> 记住这层层嵌套的关系：AI ⊃ Language AI（≈NLP）⊃ LLM。本文覆盖的历史正好对应 Language AI 发展的半个世纪。

---

## 2. 语言 AI 的发展简史：一张图看懂

语言的难点在于**非结构化**。一段文本如果用 0/1（字符级编码）表示，几乎丢失了所有语义。过去几十年 NLP 的主线，其实就是一个问题：**如何把语言转换成一种计算机能高效处理、又尽可能保留语义的结构化表示？**

![语言 AI 简史时间线（原书 Figure 1-1）](/book-notes/hands-on-llm/images/intro-history-timeline.png)

原书给出的时间线按"架构家族"给模型上色：Decoder-only（红，GPT 系）、Encoder-only（紫，BERT 系）、Encoder-Decoder（绿，T5/Switch）、Non-transformer（橙，早期模型）。这张图隐含的叙事是：**语言 AI 一直在寻找越来越好的"语义压缩方式"**，而真正的拐点是 2017 年 Transformer 的出现。

下面我们按时间顺序，看语言是如何一步步被"变得可计算"的。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/nlp-history-timeline.html" height="620px" title="NLP 演进时间线（可交互）" />

---

## 3. Bag-of-Words：最朴素的文本表示

**词袋模型（Bag-of-Words, BoW）** 大约在 1950 年代被提出、2000 年代流行起来，是 NLP 的第一代主流技术。它的想法朴素到让人惊讶：

1. **Tokenization（分词）**：把句子按空格切成一个个 token（词）。
2. **Vocabulary（建词表）**：把所有文档中出现过的 unique token 汇总成一个词表。
3. **Count（计数）**：对每个文档，统计每个词出现的次数——得到一个维度等于词表大小的向量。

![词袋模型的向量化过程（原书 Figure 1-5）](/book-notes/hands-on-llm/images/intro-bag-of-words.png)

比如有两个句子 "That is a cute dog" 和 "My cat is cute"。词表 = `{that, is, a, cute, dog, my, cat}`。"My cat is cute" 对应的向量是 `[0, 1, 0, 1, 0, 1, 1]`。

### BoW 的致命局限

BoW 有两个根本性问题：

- **丢失顺序**："dog bites man" 和 "man bites dog" 向量完全一样，但语义相反。
- **丢失语义**：它把语言看作"字面意义上的一袋词"，无法理解 "king" 和 "queen" 是相关的、"cat" 和 "dog" 都属于"宠物"这个概念。

> [!tip]
> BoW 虽然古老，但并未过时。在信息检索（TF-IDF、BM25）中它仍是主力。原书 Chapter 5 还会讲如何用它来补充现代 LLM——稀疏检索（sparse retrieval）在 RAG 中经常与语义检索配合使用。

---

## 4. Word2Vec 与 Dense Embeddings：让语义进入向量空间

BoW 的问题在 2013 年被 **word2vec** 解决了——Mikolov 等人提出用神经网络学习稠密向量（dense embedding），向量本身要捕捉**语义**。

### 核心思想：相似的词有相似的邻居

Word2Vec 利用了一个朴素的分布式假设——**出现在相似上下文里的词，语义相近**（"You shall know a word by the company it keeps"）。训练方式有两种：

| 变体 | 训练目标 |
|------|---------|
| **CBOW** (Continuous Bag-of-Words) | 用周围词预测中心词 |
| **Skip-gram** | 用中心词预测周围词 |

具体做法：给每个词随机初始化一个向量（比如 50 维）；从语料中采样 `(word, neighbor)` 词对，训练一个神经网络判断两个词是不是邻居；训练过程中不断更新它们的向量。训练完后，**神经网络的副产品——那张 embedding 表——就是每个词的向量表示**。

![Word2Vec 得到的词向量可以被解读为属性（原书 Figure 1-8）](/book-notes/hands-on-llm/images/intro-word2vec-properties.png)

上图是原书的一个经过简化的示意：每个维度仿佛代表一个"属性"（动物性、新生、人类、复数、水果……），向量就是这些属性的取值组合。**真实 embedding 的维度并不对应具体概念，但这个直觉是对的**——embedding 把离散的词映射到了一个连续的向量空间，语义相似的词在空间中彼此靠近。

### BoW vs Word2Vec 对比

| 维度 | Bag-of-Words | Word2Vec |
|------|-------------|----------|
| 向量类型 | 稀疏 (sparse) | 稠密 (dense) |
| 维度 | = 词表大小（数万到数十万） | 固定（50~300 维） |
| 语义 | 无 | 有（相似词向量接近） |
| 粒度 | 文档级 | 词级 |
| 训练 | 无需训练 | 神经网络训练 |
| 缺陷 | 丢顺序、丢语义 | **一词一向量**，无法处理多义词 |

> [!warning]
> Word2Vec 的向量是 **static（静态的）**。"bank" 这个词在 "river bank"（河岸）和 "bank account"（银行账户）里语义完全不同，但 word2vec 给它的是同一个向量。这个问题要等到 Contextual Embedding 出现才被解决。

---

## 5. Attention：让模型学会"看向哪里"

要解决静态 embedding 的问题，模型需要能够**根据上下文动态调整词的表示**。这一步最初是通过 **RNN（循环神经网络）** 实现的。

RNN 把序列一个词一个词地送进去，维护一个"隐藏状态"来记住前面看到的内容。在机器翻译任务里，一个典型架构是 **Encoder-Decoder**：

- **Encoder RNN** 读完整个输入句子，输出一个"context embedding"（上下文向量），代表整句话的意思。
- **Decoder RNN** 基于这个 context embedding，自回归地（autoregressive）生成目标语言的句子。

问题来了：整个输入句子被压缩成**一个**固定长度的 context 向量。句子越长，信息损失越严重。

### 2014：Attention 机制登场

Bahdanau 等人在 2014 年提出了 **Attention**（注意力）机制：生成每个输出词时，不只看压缩后的 context，还能**回头看输入序列的所有隐状态**，给每个输入位置打一个 attention 权重，按权重加权求和。这个权重本身是模型学出来的。

![Attention 权重矩阵：颜色越深代表两个词关联越强（原书 Figure 1-14）](/book-notes/hands-on-llm/images/intro-attention-heatmap.png)

上图是一个翻译例子（英译荷兰语）的 attention 可视化：输出 "lama's" 时，模型把大部分注意力放在输入端的 "llamas" 上——这正是人类翻译时会做的对齐。

> [!note]
> Attention 的精髓：**不是"记住一切"，而是"学会选择性地关注"**。这个 selective attention 的想法，后来成了 Transformer 的理论基石。

### RNN + Attention 的局限

RNN 即使加上 attention，仍然是**顺序处理**的——必须一个 token 一个 token 地算，没法并行。这对训练速度是个硬伤，尤其当语料规模上到整个互联网时。

---

## 6. 2017：Attention Is All You Need — Transformer 登场

Vaswani 等人在 NeurIPS 2017 发表的 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 提出了划时代的 **Transformer** 架构。它的核心主张：**既然 attention 这么有用，不如彻底扔掉 RNN，全用 attention 搭建网络**。

### Transformer 的关键突破

![Transformer：堆叠的 encoder + decoder（原书 Figure 1-16）](/book-notes/hands-on-llm/images/intro-transformer-stacked.png)

原始 Transformer 是 **Encoder-Decoder** 结构，但每个 encoder/decoder block 内部都围绕 **self-attention** 构建：

- **Self-attention**：让序列中每个位置都可以"看到"序列里其他所有位置，并按学到的权重加权聚合。这允许模型**在一次 forward pass 中同时处理整个序列**——天然并行。
- **Feed-forward network**：每个位置独立经过一个 MLP，进一步变换表示。
- **Decoder 的 masked self-attention**：为了保证生成时不"偷看未来"，decoder 中每个位置只能 attend 到当前及之前的位置。

相比 RNN，Transformer 有两个巨大优势：

| 优势 | 为什么重要 |
|------|----------|
| **并行化训练** | 不再是一步步递推，可以铺满 GPU，训练速度数量级提升 |
| **长距离依赖** | 任意两个位置之间距离都是 O(1)，再远的依赖都能直接捕捉 |

> [!tip]
> Transformer 的并行性直接催生了"大"——没有它就没有后来的 GPT 系列。这是为什么很多人把 2017 定为 LLM 时代的起点。

Transformer 架构本身还有许多细节（multi-head attention、positional embedding、layer normalization 等），原书第 2、3 章会深入，这里先掌握"Transformer = self-attention + 并行"这个直觉。

---

## 7. 两条分支：Encoder-only vs Decoder-only

原始 Transformer 是 encoder-decoder 结构、主要用于翻译。2018 年后，研究者意识到可以**只保留一半**来适配不同任务。语言 AI 就此分叉为两大架构家族。

### BERT（2018）：Encoder-only，专注"理解"

![BERT 架构：12 层 encoder 堆叠（原书 Figure 1-21）](/book-notes/hands-on-llm/images/intro-bert-architecture.png)

**BERT**（Bidirectional Encoder Representations from Transformers）丢掉 decoder、只保留 encoder 堆叠。输入前面加一个特殊的 `[CLS]` token，它的最终 embedding 被当作整句话的表示。

BERT 的训练任务叫 **Masked Language Modeling（MLM）**：随机遮掉输入中 15% 的词，让模型预测被遮的是什么。因为 self-attention 是**双向**的（每个位置既看前面也看后面），BERT 学到的 embedding 蕴含了双向上下文信息——对"理解"型任务（分类、命名实体识别、语义搜索）非常有效。

BERT 引入了 **"预训练 + 微调"** 这种两阶段范式：先在 Wikipedia 全文上做 MLM 预训练得到通用表征，再在下游任务（如情感分类）上少量数据微调。这个范式主宰了 2018-2022 的 NLP。

### GPT（2018）：Decoder-only，专注"生成"

GPT（Generative Pre-trained Transformer）走了相反方向：丢掉 encoder、只保留 decoder。GPT-1 只有 117M 参数，在 7000 本书 + Common Crawl 上训练。

![GPT 参数量的指数级增长（原书 Figure 1-25）](/book-notes/hands-on-llm/images/intro-gpt-scaling.png)

GPT 的训练任务简单到令人发指——**预测下一个词**（next-token prediction）。由于是单向 masked self-attention，每个位置只能看前面的 token，刚好契合自回归生成。

然后 OpenAI 开始不断放大：GPT-1（117M）→ GPT-2（1.5B）→ GPT-3（175B）。参数量每代增长约 10-100 倍，涌现了**指令遵循、少样本学习（few-shot learning）**等一代模型才有的能力。

### 为什么 GPT 后来居上？

| 维度 | Encoder-only (BERT 系) | Decoder-only (GPT 系) |
|------|----------------------|----------------------|
| 代表模型 | BERT, RoBERTa, DistilBERT | GPT-1/2/3/4, Llama, Claude |
| 注意力方向 | 双向 | 单向（causal mask） |
| 擅长任务 | 分类、NER、语义搜索、Embedding | 文本生成、对话、推理 |
| 部署方式 | 预训练 + 任务微调 | 预训练 + 指令微调 + prompt |
| 一个模型能做多少任务？ | 少，每个下游任务通常要一个微调版本 | 多，同一个模型通过 prompt 即可完成几十种任务 |
| 通用性 | 较弱 | 很强 |
| 规模效应 | 放大后收益递减 | 持续涌现新能力 |

GPT 之所以赢得了这轮竞赛，本质上是**通用性**的胜利——一个足够大的生成模型可以通过 prompt 模拟绝大多数 NLP 任务；而 BERT 每接一个新任务都要重新微调。当模型足够大，生成能力反而能"吃掉"理解任务。

> [!note]
> 但请注意：原书明确指出两类都属于 LLM 范畴。Encoder-only 模型在 embedding、语义搜索、RAG 中仍是生产主力。它们并没有过时，只是不像 ChatGPT 那样走到台前。

---

## 8. 2023：生成式 AI 之年

2022 年 11 月 30 日，OpenAI 发布了 ChatGPT（基于 GPT-3.5）。5 天百万用户，2 个月一亿用户——它彻底改变了大众对 AI 的认知。

2023 年被原书作者称为 **The Year of Generative AI**。整个行业以前所未有的节奏同时发布闭源和开源模型：

![2023 生成式 AI 之年：闭源与开源模型双线爆发（原书 Figure 1-28）](/book-notes/hands-on-llm/images/intro-year-of-genai.png)

- **闭源（Proprietary）**：GPT-4、BARD、PaLM 2、Claude 2、Gemini、Grok……
- **开源（Open）**：Llama（Meta）、Falcon（TII）、MPT、Qwen（阿里）、Mistral、Yi、Mixtral 8×7B、Phi-2、DeciLM、Command R……

除了数量的爆发，更重要的是新范式：这些大模型被称为 **Foundation Models（基础模型）**——一个预训练好的通用底座，可以被微调或 prompt 到无数下游任务上。"一个模型打天下"从学术口号变成了工程现实。

此外，**Mamba**（基于选择性状态空间）、**RWKV**（把 RNN 重新适配到 Transformer 时代）等新架构也在 2023 年出现，试图在长上下文、推理速度上超越 Transformer。

---

## 9. LLM 的训练范式：预训练 + 微调

传统机器学习通常是"单步"的：针对某个特定任务（如情感分类）收集标注数据，训练出一个专用模型。

LLM 的训练至少分**两步**：

### Step 1: Pretraining（预训练）

又叫 language modeling，花费绝大多数算力。模型在海量互联网文本上做 **self-supervised learning**——对 decoder-only 是 next-token prediction，对 encoder-only 是 MLM。这个阶段目标很"宽泛"：学会语法、语境、世界常识、推理模式。

训练完成的模型叫 **base model** 或 **foundation model**。它们**不会遵循指令**——你让它"写一个关于鸡的笑话"，它可能会接着续写"……但这个要求有点奇怪，因为……"，因为它只学会了"接着写"。

举个规模感的例子：Llama 2 在 **2 万亿 token** 上训练，Meta 用了 A100-80GB GPU 集群。按 1.5 美元/小时/GPU 算，总成本超过 **500 万美元**。

### Step 2: Fine-tuning（微调 / post-training）

在 base model 的基础上，用少得多、但质量高得多的数据继续训练，把模型调教成符合某个特定目标的形态：

- **SFT（Supervised Fine-Tuning）**：用 `(指令, 理想回答)` 数据对训练，让模型学会跟随指令（指令微调）。
- **RLHF（Reinforcement Learning from Human Feedback）**：用人类对回答的偏好排序作为信号，通过强化学习进一步让模型的输出对齐人类价值观。
- **任务微调**：在具体任务数据上微调，比如情感分类、代码补全。

![LLM 的两阶段训练（原书 Figure 1-30）](/book-notes/hands-on-llm/images/intro-training-paradigm.png)

> [!tip]
> 预训练成本高到令人望而却步，但微调门槛要低得多——几块消费级 GPU、几百到几千条样本就能做。原书第 12 章专门讲如何微调基础模型。**对大多数开发者来说，预训练阶段永远是别人做，微调才是自己的工作**。

---

## 10. LLM 为什么这么有用？

原书罗列了几个典型应用，展示 LLM 的多样性：

- **监督分类**（是否正面评论）——encoder-only 或 decoder-only 都行
- **无监督聚类**（工单主题发现）——embedding + 聚类
- **语义搜索与检索**（RAG 的基础）——embedding 向量库
- **Chatbot + 工具调用**（prompt engineering + RAG + 微调的组合拳）
- **多模态**（Vision + Language，比如"看冰箱内容写菜谱"）

LLM 价值的本质来自三样东西：

1. **通用性**：一个模型能覆盖翻译、摘要、问答、写代码等几十种任务。
2. **Few-shot / In-context learning**：给几个示例 LLM 就能模仿格式。
3. **指令遵循**：经过 RLHF 的模型能跟随自然语言指令，大幅降低使用门槛。

这三者叠加，把 LLM 从"实验室玩具"变成了生产力工具。

---

## 11. 开源 vs 闭源：两种生态

原书第 1 章最后讨论了访问 LLM 的两种方式，这不只是技术选择，更是生态选择。

| 维度 | 闭源（Proprietary） | 开源（Open） |
|------|-------------------|-------------|
| 代表模型 | GPT-4、Claude、Gemini | Llama、Mistral、Phi、Command R、Qwen |
| 访问方式 | 通过 API | 下载权重本地运行 |
| 硬件要求 | 厂商托管，用户无 GPU 需求 | 需要用户自备 GPU（或租云 GPU） |
| 数据隐私 | 数据发给厂商 | 完全自己控制 |
| 微调自由度 | 通常不行或受限 | 完全自由，可定制 |
| 性能天花板 | 通常更高（重金投入） | 快速追赶中 |
| 成本模型 | 按 token 付费 | 硬件 + 电费 |

**Hugging Face** 在开源生态中扮演了关键角色——像 GitHub 之于代码、PyPI 之于 Python 包，它是 LLM 的中央仓库。截至本书写作时，Hugging Face Hub 上已有超过 **80 万** 个模型，涵盖语言、视觉、音频、表格等各种模态。原书作者也直言不讳：**"我们更倾向于使用开源模型"**——自由度和可控性的价值大于便利。

> [!warning]
> 所谓"开源"的定义其实有争议。有些模型虽然公开了权重但不允许商用；有些虽然允许商用但训练数据和代码并未公开。真正符合 OSI 开源定义的 LLM 其实不多，业界更多用"open models / open-weight"这个更精准的说法。

---

## 12. 第一个代码示例：用 Phi-3 生成文本

原书在第 1 章结尾用一段简洁代码演示了完整的"加载模型 → 生成文本"流程。用的是微软的 **Phi-3-mini-4k-instruct**——3.8B 参数，小到可以在 8GB VRAM 的消费级 GPU 上跑。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 封装成 pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

# 用 chat 格式组织 prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]
output = generator(messages)
print(output[0]["generated_text"])
# Why don't chickens like to go to the gym? Because they can't crack the egg-sistence of it!
```

几个值得注意的参数：

- `return_full_text=False`：只返回模型生成的部分，不带 prompt。
- `max_new_tokens=500`：限制生成长度，防止模型无节制地输出直到 context window 满。
- `do_sample=False`：贪婪解码——每次都选概率最高的下一个 token。设为 `True` 并配合 temperature 可以让输出更有创造性，原书第 6 章会详细讨论采样策略。

使用 LLM 时加载的是**两个模型**：生成模型本身 + tokenizer（负责把文本切成 token ID，再把生成的 ID 转回文本）。Tokenizer 和模型必须**配对使用**——这个细节会在第 2 章深入。

---

## 13. 本章核心概念速查表

| 概念 | 作用 / 定位 | 进一步阅读 |
|------|-----------|-----------|
| **Language AI** | AI 中处理语言的子领域，约等于 NLP + 检索等 | 本章 §1 |
| **Bag-of-Words** | 最朴素文本表示，统计词频。丢序、丢义 | §3，原书 Ch5 |
| **Word2Vec** | 2013，稠密语义向量。解决 BoW 的语义问题，但一词一向量 | §4，原书 Ch2 |
| **RNN + Attention** | 2014，让 seq2seq 能处理长句 | §5 |
| **Transformer** | 2017，纯 attention 架构，支持大规模并行训练 | §6，原书 Ch3 |
| **BERT (Encoder-only)** | 2018，双向 MLM 预训练，专于理解任务 | §7，原书 Ch4, Ch11 |
| **GPT (Decoder-only)** | 2018 起，单向 next-token 预测，专于生成任务 | §7，原书 Ch3 |
| **Foundation Model** | 大规模预训练的通用底座，通过微调/prompt 适配下游 | §8 |
| **Pretraining** | 自监督、大算力阶段，学通用语言能力 | §9 |
| **Fine-tuning (SFT/RLHF)** | 小算力、有监督、对齐人类意图 | §9，原书 Ch12 |
| **Context length** | LLM 能处理的最大 token 数，直接决定可输入文档长度 | §10 |
| **Open vs Proprietary** | 权重是否公开、能否本地部署 | §11 |
| **Hugging Face** | 开源 LLM 的"GitHub"，模型/数据/代码的中央仓库 | §11 |

---

## 延伸思考

这一章虽然是"导论"，却埋下了全书的几条主线：

1. **表征（representation）永远是核心问题**。从 BoW 的词频向量，到 word2vec 的稠密 embedding，到 Transformer 的 contextual embedding——语言 AI 的每一次跃迁都是"表示变得更好"。
2. **架构选择决定任务适配**。encoder-only 和 decoder-only 的分野不是技术偶然，而是反映了"理解 vs 生成"两类任务的本质差异。
3. **规模带来涌现**。从 GPT-1（117M）到 GPT-3（175B）的 1500 倍增长，不只是线性提升——是指令遵循、few-shot、推理等全新能力的涌现。这也是 scaling law 讨论的起点。
4. **开源正在追赶闭源**。2023 的时间线上，Llama 2、Mixtral 等模型已经在许多任务上接近 GPT-3.5 / GPT-4。开源化的速度，决定了 LLM 作为基础设施的普及速度。

接下来两章，原书将深入两个常被低估但至关重要的主题——**Tokenization 与 Embeddings**（第 2 章）、**Transformer 内部机制**（第 3 章）。那是从"会用 LLM"到"理解 LLM"的关键一跃。
