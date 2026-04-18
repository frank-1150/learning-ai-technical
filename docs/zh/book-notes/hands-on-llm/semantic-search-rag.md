---
date: 2026-04-17
title: "语义搜索与 RAG：Dense Retrieval、Reranker 与生成式问答"
description: "当 LLM 记不住你的私有数据，RAG 就是那座桥梁。系统梳理 Dense Retrieval 的原理、为什么需要 Reranker、RAG 的完整管线与评估方法，以及那些真正落地时会遇到的坑。"
tags: [RAG, 语义搜索, Dense Retrieval, Reranker, Embedding]
---

# 语义搜索与 RAG：Dense Retrieval、Reranker 与生成式问答

> 本文对应原书 **第 8 章 Semantic Search and Retrieval-Augmented Generation**，覆盖：语义搜索与关键词搜索的对比、Dense Retrieval、Reranker、RAG 管线、评估指标、高级 RAG 技术。

搜索是 language model 最早被工业界大规模采用的应用之一。BERT 发布几个月后，Google 就宣布用它驱动搜索，称之为"搜索史上最大的一次跃进"；Bing 紧随其后，也表示 transformer 给 Bing 带来了过去一年最大的质量提升。**语义搜索（semantic search）**——按含义而不是按关键字匹配——从那时起成为 LLM 时代搜索系统的标配。

与此同时，人们开始把生成式模型当成问答引擎，却很快撞上"幻觉"这堵墙：模型答得很流利，但内容经常不对、不新。为此，工业界普遍采用的救赎方案是 **RAG（Retrieval-Augmented Generation，检索增强生成）**——把"相关资料"先检索出来，再塞给 LLM 去组织答案。本文用问题驱动的方式，把 Dense Retrieval、Reranker、RAG 管线和评估这几块内容串起来。

## 一、动机：为什么我们需要 RAG

LLM 虽然"见多识广"，但有三个硬约束决定了它无法独立回答企业场景的大部分问题：

1. **知识截断（knowledge cutoff）**：模型训练数据有时间边界。GPT-4o 的知识截止到 2023 年 10 月，你问它 2024 年的事，它要么说不知道，要么编。
2. **幻觉（hallucination）**：对于训练语料里见过但不太熟的内容，模型会"自信地编造"——这是生成式模型的底层特性，无法靠 prompt 彻底消除。
3. **私有数据盲区**：你公司的产品手册、内部 wiki、CRM 数据，模型训练时根本没见过。

一个朴素的想法是："直接把文档塞进 context 不就行了？"——但这条路走不远：

- **上下文长度**：即使有 128K/200K token 的超长 context，一个中型企业的知识库几十 GB，全塞根本塞不下。
- **成本**：API 按 token 计费，每次请求都塞 50K token 的资料，单价上涨几十倍，延迟也会劣化。
- **精度下降**：研究表明 LLM 在超长上下文里存在 "lost in the middle" 现象，放在中间的关键信息经常被忽略。

于是 RAG 的基本思路呼之欲出：

> **先用搜索找到问题最相关的几个片段（chunks），只把这些片段放进 prompt，让 LLM 基于它们生成答案。**

这套管线把"记忆"和"推理"解耦——LLM 专注做它擅长的事（理解、组织、生成），把"查资料"交给一个专门的检索系统。

![RAG 系统架构](/book-notes/hands-on-llm/images/rag-architecture.png)

书中把搜索相关的 LLM 应用分成三大类：**Dense Retrieval（密集检索）**、**Reranking（重排）**、**RAG（检索增强生成）**。前两者是搜索系统本身的升级，第三者是在搜索之上叠加生成。下文依次拆解。

## 二、语义搜索 vs 关键词搜索：BM25 的局限

在 dense retrieval 出现之前，工业界标准的搜索算法是 **BM25**——一种基于"精确 term 匹配 + TF-IDF 加权"的 lexical search。它简单、可解释、对精确短语匹配效果极好，但有三个典型短板：

- **同义词问题**：查询 "car" 匹配不到内容里写 "automobile" 的文档。
- **语义等价的不同表述**：问 "how precise was the science"，相关段落可能根本不含 "precise" 和 "science" 这两个词。
- **多语言 / 跨域**：同一概念在不同语言里字面完全不同，BM25 无能为力。

Dense retrieval 的解决思路是把 query 和 document 都映射到同一个向量空间，然后按向量距离（通常是余弦距离）找最近邻。**embeddings 让"含义相近"变成"空间距离近"**——于是 "car" 和 "automobile" 的向量可以很接近，即便它们字面毫无交集。

下面是书中的直观实验：用 Cohere 的 embedding API 对 Interstellar（星际穿越）的 Wikipedia 页面做 dense retrieval，查询 `how precise was the science`：

```python
# Dense retrieval 命中
0  It has also received praise from many astronomers for its
   scientific accuracy and portrayal of theoretical astrophysics
   distance = 10757.38
```

而同样的查询用 BM25：

```python
# BM25 命中（按词面分数排序）
1.789  Interstellar is a 2014 epic science fiction film ...
1.373  Caltech theoretical physicist ... Kip Thorne ...
0.000  It stars Matthew McConaughey ...
```

BM25 因为查询里有 "science"，把不相关的电影简介排到了第一位；而 dense retrieval 直接命中了真正讨论"科学准确性"的那一句，尽管它根本没出现 "precise" 这个词。

**BM25 vs Dense Retrieval 对比**：

| 维度 | BM25 (lexical) | Dense Retrieval (semantic) |
|------|----------------|----------------------------|
| 匹配依据 | 精确 term + TF-IDF | 向量空间的语义相似 |
| 同义词处理 | 做不到 | 天然支持 |
| 精确短语检索 | 极强（SKU、型号、引用） | 较弱，容易"语义漂移" |
| 可解释性 | 高（得分能拆到每个 term） | 低（黑盒 embedding） |
| 跨语言 | 无法 | 用多语言 embedding 可做 |
| 训练成本 | 零 | 需要 embedding model |
| 未登录域表现 | 稳定 | 在未训练的领域（如法律文书）可能失效 |

:::tip 为什么实际系统往往用 Hybrid 检索
纯 dense retrieval 在"查一个精确短语"（比如产品型号 `M1 Max 14-inch`）时可能翻车——向量漂移会把它归到"苹果笔记本"这个大类里。工业界常见做法是把两路召回合并：BM25 负责精确匹配，dense 负责语义联想，然后用倒排融合或 reranker 统一排序。Elasticsearch、Vespa、Weaviate 都原生支持 hybrid。
:::

## 三、Dense Retrieval：架构与实操

![Dense Retrieval 基本流程](/book-notes/hands-on-llm/images/rag-dense-retrieval.png)

Dense retrieval 的架构可以拆成 **离线索引** 和 **在线查询** 两个阶段。

### 离线：把所有文档编码进向量库

1. **Chunking（切分）**：把长文档切成若干小段。句子级最细，段落级更合理，3–8 句一个 chunk 是常见区间。
2. **Embedding**：用 encoder model（比如 `text-embedding-3`、`bge-small-en-v1.5`、Cohere embed）把每个 chunk 转成一个固定维度的向量（几百到几千维）。
3. **Indexing**：把所有向量存进专门优化过的向量库——FAISS（Meta，单机）、Pinecone/Weaviate/Milvus（云 / 分布式）。这些系统为 **ANN（Approximate Nearest Neighbor）** 搜索做了大量优化，能在毫秒级从百万量级向量中找出 Top-K。

![文档到向量库的转化](/book-notes/hands-on-llm/images/rag-chunking-vectordb.png)

### 在线：query 也走一遍 embedding

```python
import cohere, numpy as np, faiss
co = cohere.Client(api_key)

# 1. 把文档块嵌入
texts = [...]  # 已经切好的句子/段落列表
embeds = np.array(co.embed(texts=texts,
                            input_type="search_document").embeddings)

# 2. 建 FAISS 索引
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.float32(embeds))

# 3. 查询
def search(query, k=3):
    q_emb = co.embed(texts=[query],
                     input_type="search_query").embeddings[0]
    distances, ids = index.search(np.float32([q_emb]), k)
    return [(texts[i], d) for i, d in zip(ids[0], distances[0])]

search("how precise was the science")
```

注意 `input_type` 这个参数——Cohere、OpenAI 的 embedding API 现在都区分 **search_document** 和 **search_query**，即同一个模型内部为 query 和 doc 用略微不同的投影。这是双塔/Siamese 架构的一个优化：query 和 doc 天然不对称（query 短、doc 长），分开编码能拿到更好的对齐效果。

:::warning Dense Retrieval 的两大陷阱
**1. 训练域外表现糟糕**：在 Wikipedia + 互联网语料上训练的 embedding 模型，直接拿去做法律文书检索，效果可能不如 BM25。此时要么用领域数据 fine-tune，要么叠加 BM25 做 hybrid。

**2. 相关 ≠ 相似**：query "What is the mass of the moon?" 在一个只讲 Interstellar 的语料库里也会返回向量距离最近的几个句子，但它们跟月球质量完全无关。实践中要设一个 **距离阈值**，低于阈值的结果干脆不返回，或者把"没找到相关内容"这件事告诉用户。
:::

### Chunking：最被低估的工程决策

Chunking 策略直接决定召回质量。书中给出了三种常见切法：

| 策略 | 粒度 | 优点 | 缺点 |
|------|------|------|------|
| Character split（按字符数） | 很细 | 实现简单 | 可能从单词中间切开，破坏语义 |
| Sentence split（按句子） | 细 | 语义完整 | 上下文可能跨句，单句信息不够 |
| Paragraph split（按段落） | 中 | 语义最完整 | 单 chunk 可能太大，embedding 压缩损失 |
| **Overlapping chunks**（滑动窗口） | 可调 | 上下文连续，避免信息被切断 | 存储翻倍 |

实践中的经验法则：**约 200–500 token 一个 chunk，10–15% 的 overlap**。更进阶的做法是在 chunk 里附加"上下文标签"——比如把文档标题、所在章节标题拼进 chunk 文本再做 embedding，这样孤立的 chunk 也能获得它原本的语境。

## 四、Reranker：为什么检索要分两段

Dense retrieval 的 bi-encoder（双塔）架构有个本质短板：**query 向量和 doc 向量是分开计算的**，两者从未在模型内部"见面"，只在最后用余弦距离粗略比较。优点是快——100 万文档的 embedding 可以离线算一次然后反复用。缺点是精度上限有限。

**Cross-Encoder**（交叉编码器）是另一种思路：把 `[query, doc]` 拼在一起作为一个输入喂给 BERT 风格的模型，模型在 attention 层里让 query 和 doc 的 token 互相交互，最后输出一个相关性得分（0 到 1）。这种方式精度高得多——研究里常报告 nDCG@10 从 bi-encoder 的 36.5 涨到 cross-encoder 的 62.8，幅度夸张。

但 cross-encoder 慢：每个候选 doc 都要走一次完整的 forward pass，无法离线预计算。对 100 万文档跑一遍 cross-encoder，单次查询要几秒到几十秒，完全不能上线。

![Reranker 两段式管线](/book-notes/hands-on-llm/images/rag-reranker-pipeline.png)

### 工业界的标准答案：两段式检索

> **第一阶段**（粗筛）：用 BM25 或 dense retrieval 快速从百万级文档中捞出 Top-100。
>
> **第二阶段**（精排）：把 Top-100 交给 cross-encoder reranker 重新打分，挑出 Top-3 ~ Top-10。

![Cross-Encoder Reranker 打分](/book-notes/hands-on-llm/images/rag-reranker.png)

这样就把快（bi-encoder）和准（cross-encoder）两件事解耦了——粗筛快速把候选集从百万降到百级，精排虽然慢但只处理 100 个候选，单次查询总耗时可控。

**Bi-Encoder vs Cross-Encoder 对比**：

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| 输入 | query 和 doc 分别编码 | `[query; doc]` 一起编码 |
| Query 和 Doc 的交互 | 仅在最后距离计算 | 在 attention 每一层 |
| 相关性建模能力 | 中 | 强 |
| 能否离线预计算 doc 向量 | 能 | 不能 |
| 适合规模 | 百万级以上 | 百级候选 |
| 典型模型 | MiniLM、BGE、Cohere embed | monoBERT、Cohere Rerank |
| 在搜索管线的位置 | 第一阶段（recall） | 第二阶段（precision） |

使用 Cohere 的 Rerank API 非常简单，不需要训练：

```python
query = "how precise was the science"
results = co.rerank(query=query, documents=texts, top_n=3,
                    return_documents=True)
for r in results.results:
    print(r.relevance_score, r.document.text)
# 0.1698  It has also received praise ... for its scientific accuracy ...
# 0.0700  The film had a worldwide gross ...
# 0.0044  Caltech theoretical physicist ... Kip Thorne ...
```

本地方案可以用 **sentence-transformers** 里的 `CrossEncoder`，常见模型是 `ms-marco-MiniLM-L-6-v2`，百毫秒级打分一个 query-doc 对。

## 五、检索评估指标：什么叫"搜得好"

要优化检索，先要定义什么是好。IR（Information Retrieval）领域经过几十年沉淀，核心指标有：

![MAP 计算示意](/book-notes/hands-on-llm/images/rag-metrics.png)

| 指标 | 公式直觉 | 考察点 | 什么时候用 |
|------|---------|--------|------------|
| **Recall@K** | Top-K 里包含的相关文档 / 所有相关文档 | 召回率 | 候选池规模、粗筛阶段 |
| **Precision@K** | Top-K 里的相关文档 / K | 准确率 | 精排、用户只看前几条 |
| **MRR**（Mean Reciprocal Rank） | 对每个 query 取第一个相关结果位置的倒数，然后平均 | 第一个对的结果出现得早不早 | 只需要"一条正确答案"的场景（FAQ、QA） |
| **MAP**（Mean Average Precision） | 每个 query 算 AP（在每个相关文档的位置算 precision 再平均），所有 query 取平均 | 整体相关文档的排序质量 | 经典 IR 基准 |
| **nDCG@K** | 带位置折扣的 DCG 归一化到 [0, 1] | 文档的分级相关性 + 排名位置 | 相关性有等级之分时（完美/不错/可以/不相关） |

**一个极简的直觉化例子**。假设某个 query 有 1 个相关文档：

- System A 把它放在第 1 位 → Reciprocal Rank = 1/1 = 1.0
- System B 把它放在第 5 位 → Reciprocal Rank = 1/5 = 0.2

MRR 说 A 更好。这与直觉吻合：用户只看前几条，排第 5 的相关结果基本等于没有。

**nDCG 的额外价值**是处理分级相关性——"完美匹配"应该比"勉强相关"得分高得多。它引入位置折扣项 `1/log2(rank+1)`，让排在第 1 位的权重远大于第 10 位。

下文用一个交互式可视化展示同一组检索结果在不同 K 下的指标差异：

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/retrieval-metrics.html" height="500px" title="检索评估指标交互计算" />

## 六、RAG 完整管线：从搜索到生成

把前面的砖块拼起来，一个典型的 RAG 系统包含 5 个阶段：

1. **Chunking**：离线把文档切成 chunk。
2. **Indexing**：把 chunk 向量化存入向量库。
3. **Retrieval**：query 来了先做向量检索（可选混合 BM25）。
4. **Reranking**（可选）：cross-encoder 精排。
5. **Generation**：把 Top-K chunks 填入 prompt 模板，交给 LLM 生成答案。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/rag-pipeline.html" height="650px" title="RAG 完整管线动画" />

### 用 LLM API 做 grounded generation

许多 LLM API（如 Cohere 的 `co.chat`）支持把 documents 作为一等公民传入，模型会在输出里自动标注引用：

```python
query = "Income generated by the film"

# 1. Retrieval
results = search(query)  # 上面定义的 dense retrieval
docs_dict = [{'text': t} for t in results['texts']]

# 2. Grounded Generation
response = co.chat(message=query, documents=docs_dict)

print(response.text)
# The film generated a worldwide gross of over $677 million,
# or $773 million with subsequent re-releases.

print(response.citations)
# [ChatCitation(start=21, end=36, text='worldwide gross',
#               document_ids=['doc_0']), ...]
```

带引用（citation）的生成是 RAG 非常重要的产品能力——**用户可以点进去验证原文**。这不仅增强了可信度，也为"模型答错时把锅甩给数据而不是模型"提供了审计依据。

### 用本地模型跑 RAG

如果不想依赖外部 API，用 LangChain + LlamaCpp + FAISS + HuggingFace embedding 可以完整本地化：

```python
from langchain import LlamaCpp, PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 本地生成模型（量化）
llm = LlamaCpp(model_path="Phi-3-mini-4k-instruct-fp16.gguf",
               n_gpu_layers=-1, max_tokens=500, n_ctx=2048)

# 2. 本地 embedding 模型（MTEB leaderboard 上的小模型）
embedding = HuggingFaceEmbeddings(model_name='thenlper/gte-small')

# 3. FAISS 向量库
db = FAISS.from_texts(texts, embedding)

# 4. Prompt 模板
template = """<|user|>
Relevant information:
{context}

Provide a concise answer to the following question using the
relevant information provided above:
{question}<|end|>
<|assistant|>"""
prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# 5. 串起来
rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt})

print(rag.invoke("Income generated"))
```

**Prompt 模板的关键设计要点**：

- 明确告诉模型 "use the relevant information provided above"，否则它可能忽略 context 用自己的先验知识作答。
- 用清晰的结构分隔符（`<|user|>`、`<|end|>`、`<|assistant|>`）让模型识别角色边界——不同模型格式不同，Phi-3 用 `<|user|>`，Llama-3 用 `<|start_header_id|>user<|end_header_id|>`，ChatML 用 `<|im_start|>`。
- 没检索到结果时的 fallback：要显式引导模型说 "I don't know based on the provided information"，而不是让它硬编。

## 七、高级 RAG 技术：当基础版不够用

基础版 RAG 在简单问答上够用，但面对复杂 query 会暴露很多问题：query 太啰嗦、跨多文档、需要多步推理……书中列出了几种常见的增强手段。

### Query Rewriting（查询重写）

原始 query 可能上下文冗余，直接去检索召回很差。让 LLM 先把它重写成一个"适合检索的版本"：

```
User: "We have an essay due tomorrow. We have to write about some animal.
      I love penguins. I could write about them. But I could also write
      about dolphins. Are they animals? Maybe. Let's do dolphins.
      Where do they live for example?"

Rewritten query: "Where do dolphins live"
```

### HyDE（Hypothetical Document Embeddings）

思路反直觉但有效：**先让 LLM 直接回答问题（生成一个假设的答案文档）**，然后用这个假设文档的 embedding 去检索，而不是用原始 query 的 embedding。

原因是 query 和 doc 的语言风格经常差异很大（query 短、像问题；doc 长、像陈述），embedding 空间里未必相近。让 LLM 生成一个"像 doc 的东西"再去检索，能缩小这个 query-doc gap。

### Multi-Query RAG

有些问题必须拆成多个子查询才能答：

```
User: "Compare the financial results of Nvidia in 2020 vs. 2023"

Queries:
  Query 1: "Nvidia 2020 financial results"
  Query 2: "Nvidia 2023 financial results"
```

让 LLM 先判断要不要拆、拆几个，然后对每个子查询独立检索，最后把所有检索结果合并喂给 LLM 生成最终答案。

### Multi-Hop RAG

更进一步——有些问题需要 **多轮顺序检索**，后面的 query 依赖前面的结果：

```
User: "Who are the largest car manufacturers in 2023? Do they each make EVs or not?"

Step 1: search("largest car manufacturers 2023")
        → Toyota, Volkswagen, Hyundai
Step 2: 基于 Step 1 的结果拆出 3 个子查询
        search("Toyota Motor Corporation electric vehicles")
        search("Volkswagen AG electric vehicles")
        search("Hyundai Motor Company electric vehicles")
```

### Query Routing 和 Parent-Child Chunking

- **Query Routing**：给模型描述多个数据源（Notion 存 HR 信息、Salesforce 存客户数据），让它根据 query 路由到对应的源去检索。
- **Parent-Child Chunking**：小 chunk 做检索（embedding 更锐利），但返回给 LLM 时带上父段落（上下文更完整）。这是"检索精度"和"生成质量"之间的经典权衡。

所有这些技术的共同趋势是 **把更多决策从硬编码转给 LLM**——本质上就是在把 RAG 推向 **Agentic RAG**。此时 LLM 不再只是"生成答案的最后一步"，而是掌控整个检索策略、选择数据源、决定要不要再查一轮。Cohere 的 Command R+ 就是为此类任务专门设计的开源权重模型。

## 八、RAG 评估：检索好不好 ≠ 答案好不好

评估 RAG 要分两层：

### 检索质量（retrieval quality）

用前面提到的 MRR、nDCG、Recall@K——前提是你有一份 query + 相关文档的 ground truth。业界公开 benchmark 有 **MIRACL**（多语言检索）、**BEIR**（零样本 IR 基准）、**MTEB**（embedding 综合评估）。

### 生成质量（generation quality）

更棘手——生成的答案没有唯一标准。2023 年的论文 *"Evaluating verifiability in generative search engines"* 提出四个维度：

| 维度 | 定义 |
|------|------|
| **Fluency**（流畅度） | 生成的文本是否通顺、连贯 |
| **Perceived Utility**（感知效用） | 答案是否有用、提供了信息 |
| **Citation Recall**（引用召回） | 生成的事实性陈述中有多少被引用支持 |
| **Citation Precision**（引用精度） | 生成的引用中有多少确实支持了对应陈述 |

人工评估最可靠，但贵。**LLM-as-a-Judge** 是现在流行的折中——让一个强模型（GPT-4）作为"法官"给答案打分。开源框架 **RAGAS** 把这些自动化评估打包好了，核心指标包括：

- **Faithfulness**（忠实度）：答案是否完全可以从提供的 context 推出（反映幻觉程度）。
- **Answer Relevance**（答案相关性）：答案是否紧扣 query 而不跑题。
- **Context Recall / Precision**：检索到的 context 本身质量。

:::tip 落地时的评估实践
不要等系统"完成了"才开始评估。从一开始就准备一个 **~50 条 query 的 golden set**，每条标注期望的相关文档和参考答案。每次调整 chunk size、embedding model、reranker 时，都跑一遍这个集合，看指标是涨了还是跌了。

RAG 系统的改进往往是一个"指标此起彼伏"的过程——调大 chunk size 提升了 answer relevance 但降低了 retrieval recall；加 reranker 提升了 precision@3 但增加了 300ms 延迟。没有端到端的评估集，所有改动都只是瞎猜。
:::

## 九、小结

| 主题 | 关键点 |
|------|--------|
| **为什么 RAG** | 绕过知识截断、缓解幻觉、接入私有数据；直接塞 context 受限于长度/成本/精度 |
| **Dense Retrieval** | query/doc 都转 embedding，用 ANN 搜最近邻；bi-encoder 快但精度有限 |
| **Chunking** | 最被低估的工程决策；推荐 200-500 token + 10-15% overlap + 加上下文标签 |
| **BM25 vs Dense** | 前者精确 term、后者语义联想；生产系统多用 Hybrid 融合 |
| **Reranker** | cross-encoder 让 query 和 doc 在 attention 层交互，精度大幅提升但慢 |
| **两段式检索** | 第一段召回到 Top-100，第二段精排到 Top-3/Top-10，兼顾快与准 |
| **评估指标** | Recall@K 看覆盖、MRR 看第一个命中、nDCG 看分级排序质量 |
| **RAG 管线** | Chunking → Indexing → Retrieval → (Rerank) → Generation |
| **高级技术** | Query Rewriting、HyDE、Multi-Query、Multi-Hop、Query Routing、Agentic RAG |
| **RAG 评估** | 检索层看 MRR/nDCG；生成层看 Faithfulness / Citation Recall；RAGAS 做自动化 |

RAG 是 2023-2024 年 LLM 工程实践里最成熟的落地范式之一。它的精髓不在某个单点技术，而在于 **把"记忆"和"推理"解耦**——让 LLM 专注它擅长的组织与生成，把"知道什么"交给一个可维护、可更新、可审计的检索系统。理解每个环节的权衡，才能在具体业务里把它调到"刚刚好"的形态。
