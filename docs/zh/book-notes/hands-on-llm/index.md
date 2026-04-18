---
title: "Hands-On Large Language Models 阅读笔记"
description: "Jay Alammar & Maarten Grootendorst 著，O'Reilly 出版。从 Transformer 原理到 RAG、多模态与微调，系统拆解大语言模型 12 章内容。"
date: "2026-04-17"
---

# Hands-On Large Language Models 阅读笔记

![封面](/book-notes/hands-on-llm/images/cover.png)

## 书籍简介

**Hands-On Large Language Models: Language Understanding and Generation** 由 Jay Alammar 和 Maarten Grootendorst 合著，O'Reilly 于 2024 年 9 月出版（ISBN: 978-1-098-15096-9）。

两位作者都不是"纯学术"身份，而是长期在"把复杂概念讲清楚"这件事上深耕的实践者：

- **Jay Alammar** 是 Cohere 的 Director and Engineering Fellow，更广为人知的身份是 *The Illustrated Transformer*、*The Illustrated GPT-2*、*The Illustrated BERT* 等系列博客的作者。他在 ICLR 2021 Workshop 上的论文 *"Machine learning research communication via illustrated and interactive web articles"* 系统阐述了用视觉方式传达 ML 研究的方法论——这本书是这套方法论在大语言模型领域的全面落地。
- **Maarten Grootendorst** 是 IKNL（荷兰国家癌症研究中心）的高级临床数据科学家，同时也是 **BERTopic** 开源项目的作者——BERTopic 是目前主流的基于 Transformer 的主题建模框架，GitHub 星标已过万。书中第 5 章的主题建模部分正是由作者本人用他的工具从头展开。

全书贯穿的核心理念是 **"Intuition-First Philosophy"**：

> The main goal of this book is to provide an **intuition** into the field of LLMs. The pace of development in the Language AI field is incredibly fast and frustration can build trying to keep up with the latest technologies. Instead, we focus on the fundamentals of LLMs and intend to provide a fun and easy learning process.

作者明确拒绝"用数学公式堆砌"的解释路径，而是选择用**大量插图 + 可跑代码**的方式，把 Transformer、Attention、Embedding、RAG、LoRA 这些概念的"所以然"讲透。正文中有超过 300 张手绘示意图，贯穿 12 章始终。

全书分为三部分：

- **Part I — Understanding Language Models**（Ch 1–3）：回答 "How do large language models work?"
- **Part II — Using Pretrained Language Models**（Ch 4–9）：用预训练模型做分类、聚类、Prompt、RAG、多模态等下游任务
- **Part III — Training and Fine-Tuning Language Models**（Ch 10–12）：训练 Embedding 模型、微调表示模型、SFT/LoRA/DPO 微调生成模型

配套代码全部在 [HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) 仓库开源，所有示例都能在 Google Colab 免费 T4 GPU 上运行，最低显存要求 16 GB。

---

## 为什么读这本书

2024 年以后读 LLM 相关技术书，面临一个尴尬的局面：太多书在 GPT-4 前写成，已被时代淘汰；而最新的内容又零散在各大论文、博客、YouTube 视频中。**这本书的价值在于：它是 2024 年这个时间点，从"什么是 LLM"到"怎么微调一个 LLM"路径最完整、可视化质量最高的一本**。

**面向从应用层下沉到原理层的工程师**。如果你已经用过 ChatGPT API、写过几个 RAG 应用，现在想搞清楚：
- Tokenizer 到底怎么切词？不同模型的 tokenizer 为什么差别那么大？
- KV Cache 到底缓存了什么？RoPE 位置编码比绝对位置编码好在哪？
- Dense Retrieval 和关键字搜索背后的几何意义是什么？Reranker 为什么要用 cross-encoder？
- LoRA 凭什么只训 1% 的参数就能和全量微调打平？QLoRA 的 4-bit 量化为什么不会崩？
- DPO 和 RLHF 在数学上的等价关系是什么？

——这本书都给出了 **"用图讲清楚 + 用代码验证"** 的答案。

**图解是真正的"第一生产力"**。Alammar 的插图不是装饰，而是心智模型的载体。一张 KV Cache 的动图胜过三页公式；一张 Transformer Block 的拆解图让你一眼看清 Attention 和 FFN 的数据流。这种呈现方式极其适合建立长期记忆。

**代码基于 Hugging Face 生态**。从 `transformers`、`sentence-transformers` 到 `peft`、`trl`、`bitsandbytes`——这套工具链是目前开源 LLM 工程实践的事实标准。读完书也就顺便掌握了 HF 生态。

**覆盖面近乎完整的一张路线图**：
- 底层：Tokenization、Embedding、Transformer 架构
- 中层：Prompt Engineering、CoT、LangChain、Agent
- 上层：Semantic Search、RAG、多模态（CLIP / BLIP-2）
- 训练：对比学习 SBERT、BERTopic、SetFit、NER、SFT、LoRA、QLoRA、RLHF、DPO

换言之，读完这本书，你对 2024 年 LLM 工程实践的技术版图就有了完整的心智地图。

---

## 全书文章总览

本系列共 10 篇阅读笔记，按主题聚合而非严格按章节拆分：

| 文章 | 覆盖章节 | 核心主题 |
|------|---------|---------|
| [LLM 基础与简史](./intro-and-history) | Ch 1 | LLM 是什么、NLP 简史、Bag-of-Words → Word2Vec → Transformer 的演进 |
| [Tokens 与 Embeddings](./tokens-embeddings) | Ch 2 | 分词器内部、Token/词/文档 Embedding、Word2Vec 对比学习、推荐系统 |
| [Transformer 架构内部](./transformer-internals) | Ch 3 | 前向传播、Attention、KV Cache、RoPE、近期架构改进 |
| [文本分类与聚类](./classification-clustering) | Ch 4–5 | Representation vs. Generative 分类、BERTopic 主题建模 |
| [Prompt Engineering 与推理技巧](./prompt-engineering) | Ch 6 | Prompt 基本要素、CoT、Self-Consistency、ToT、输出校验 |
| [LangChain 链、记忆与 Agent](./langchain-chains-agents) | Ch 7 | Chain、Memory（Buffer/Window/Summary）、ReAct Agent |
| [语义搜索与 RAG](./semantic-search-rag) | Ch 8 | Dense Retrieval、Reranker、RAG 构建与评估 |
| [多模态大模型](./multimodal-llms) | Ch 9 | Vision Transformer、CLIP、BLIP-2、多模态对话 |
| [训练 Embedding 与微调表示模型](./embedding-representation-training) | Ch 10–11 | SBERT、对比学习、SetFit、NER 微调 |
| [微调生成模型 SFT/LoRA/DPO](./fine-tuning-generation) | Ch 12 | 全量微调、PEFT、LoRA、QLoRA、奖励模型、RLHF、DPO |

---

## 阅读方式说明

每篇笔记并非章节摘要，而是以**工程问题**为线索重组书中内容：先建立"为什么需要这个机制"的问题背景，再展开书中的设计思路与图解，最后落到可运行的代码示例和典型使用场景。

配套的交互式可视化覆盖了书中几个最具代表性的结构——Transformer Forward Pass、KV Cache 时序、RAG 检索链路、LoRA 权重分解——用它们补充文字难以呈现的动态信息。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/book-structure.html" height="600px" title="全书结构导航" />
