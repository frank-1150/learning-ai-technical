---
date: 2026-03-28
title: "OpenViking：用文件系统重新定义 Agent 的记忆与检索"
description: 解读字节跳动 OpenViking 的核心设计——为什么用文件系统范式来管理 Agent 上下文？L0/L1/L2 分层检索如何同时提升准确率、降低 Token 消耗？
tags: [agent, rag, memory, retrieval, context-management, openviking]
---

# OpenViking：用文件系统重新定义 Agent 的记忆与检索

> 源码：[github.com/volcengine/OpenViking](https://github.com/volcengine/OpenViking)（字节跳动 Volcengine，2026 年 1 月开源，Apache 2.0）

---

## 一、Agent 的记忆问题

在讨论 OpenViking 之前，先想一个问题：**一个持续运行的 AI Agent 需要什么样的"记忆"？**

这不是一个小问题。主流的 RAG 系统（把文档向量化，检索时做相似度搜索）在简单问答场景下够用，但放到真实的 Agent 工作流里，很快会暴露几个核心矛盾：

**1. 上下文来源分散**

Agent 需要的上下文来自多个地方——用户历史（记忆）、工具产出的文档（资源）、掌握的技能（Prompt 模板和工具调用模式）。传统架构里这些东西分散在向量数据库、关系数据库、Prompt 文件夹各处，Agent 要"想起来"一件事，需要在多个系统里分别查询，逻辑割裂。

**2. Token 消耗爆炸**

最直接的暴力解法是"把检索到的相关文档全部塞进 Context"。但文档一旦多、长，Token 数量急剧膨胀。一个长期运行的 Coding Agent 在处理复杂任务时，相关代码文件、历史对话、技术文档加起来轻松超过 100K Token——这不仅昂贵，还因为注意力机制的稀释而降低了质量。

**3. 检索质量差**

平铺的向量搜索缺乏层次感。搜"如何优化这段代码"，检索到的可能是 20 个零散代码片段，而不是"先定位到这是哪个模块的问题，再找这个模块的核心逻辑"。人类专家解决问题是自顶向下的，RAG 是自底向上的暴力匹配。

**4. 记忆不会自我更新**

Agent 完成一个任务后学到了什么？什么都没学到。每次对话都是全新的起点，之前解决过的问题、用户的偏好、踩过的坑——全部消失。

OpenViking 的核心论点是：**这些问题的根源是存储范式错了**。

---

## 二、核心思路：文件系统范式

OpenViking 的解法出乎意料地直觉化——它把所有 Agent 上下文，无论是记忆、文档还是技能，统一组织成一个**虚拟文件系统**，通过 `viking://` URI 协议访问。

```
viking://
├── resources/              # 账户级共享文档/数据
├── user/{space}/           # 用户私有记忆（访问隔离）
│   ├── profile/            # 用户偏好、背景信息
│   └── history/            # 历史对话摘要
├── agent/{space}/          # Agent 专属空间
│   ├── skills/             # 技能：Prompt 模板、工具调用模式
│   └── memories/           # Agent 的经验积累
└── session/{id}/           # 会话级临时上下文
    ├── context/            # 本次任务的相关资料
    └── history/            # 本次对话记录
```

Agent 与这个文件系统的交互方式和 Unix shell 完全一致：`ls`、`find`、`grep`、`tree`，甚至提供了 `ov` 命令行工具。

这个设计选择有几个深层含义：

**统一的心理模型**：记忆、资源、技能不再是三个不同的系统——它们都是"文件"，放在不同"目录"里。Agent 只需要学会"查文件"，不需要学三套 API。

**天然的层次结构**：文件系统本身就是树形的。这为后面的层次化检索提供了物理基础——"先找对目录，再找对文件"，而不是在所有文件里暴力搜索。

**隔离与共享并存**：`resources/` 是全局共享的，`user/` 是用户私有的，`session/` 是临时的。访问控制天然映射到目录权限。

**可调试性**：文件系统路径是可读的。你能看到 Agent 从 `viking://agent/coding/skills/refactoring.md` 里读取了什么，而不是"从向量数据库取回了 5 个 chunk"。

---

## 三、L0 / L1 / L2 分层检索：解决 Token 爆炸

文件系统解决了"放哪里"的问题，但没解决"取多少"的问题。OpenViking 的第二个核心设计是**语义层次化存储**。

每个被写入 OpenViking 的文件，都会被 `SemanticProcessor` 自动生成三个语义层级：

| 层级 | Token 预算 | 生成方式 | 用途 |
|---|---|---|---|
| **L0 摘要** | ~100 tokens | VLM 生成的极简摘要 | 向量索引 + 目录级过滤 |
| **L1 概览** | ~2,000 tokens | 层次化结构化摘要 | 重排序 + 导航决策 |
| **L2 详情** | 无限制 | 原始内容 | 需要时才按需加载 |

检索过程是分层的：

```
查询: "优化 Python 代码中的内存泄漏"
    ↓
1. 对查询做语义扩展，生成 3~5 个子查询
   ["内存泄漏检测", "Python gc模块", "弱引用 weakref", ...]
    ↓
2. 对所有文件的 L0 摘要做向量搜索（廉价）
   → 命中 3 个高分目录
    ↓
3. 在命中目录内做递归搜索
   → 命中具体文件的 L0，用 L1 做重排序
    ↓
4. 仅对最终需要的文件加载 L2 详情
```

这个设计的关键洞察是：**大多数检索任务在 L0 层就能做出导航决策，L2 只在真正需要细节时才付出代价**。

和传统 RAG 对比：传统 RAG 在检索时把文档切成固定大小的 chunk，然后无差别地向量化所有 chunk。这导致两个问题：一是语义被人为截断（一段代码注释和它对应的函数被切开了），二是检索时不论最终用不用，都要把完整 chunk 塞进 Context。

L0/L1/L2 的分层让 Token 消耗变成了按需的——只有确定要读的文件，才加载它的完整内容。

---

## 四、目录递归检索：像人类专家一样思考

OpenViking 的检索算法有一个精妙的设计：**分数从子节点向父目录传播**。

具体来说，当一个文件的 L0 摘要在向量搜索中得分很高时，它的父目录也会获得一个衍生分数：

```
directory_score = 0.5 × parent_directory_score + 0.5 × embedding_similarity
```

这个公式让检索算法能做出一个传统 RAG 做不到的判断：**"这个目录里有很多高相关文件，所以这个目录本身就是相关的，值得在它内部做更深入的搜索"**。

这模拟的是人类专家在一个大型代码库里找相关代码的过程：先判断"这是个基础设施相关的 bug，去 `infra/` 目录"，再判断"具体是网络问题，进 `infra/networking/`"，最后才打开具体文件。

这个递归过程可以配置深度和广度，最终返回的不是一堆散乱的文档片段，而是一个**有层次的、有路径信息的检索结果树**。

---

## 五、会话结束后的自进化记忆

每次对话结束后，OpenViking 会自动执行一次"记忆巩固"流程：

```
对话结束
    ↓
1. 完整对话 + 使用的上下文归档到 session/{id}/history/
    ↓
2. VLM 从对话中提取结构化记忆片段
   - 用户层面：偏好、工作背景、常用模式
   - Agent 层面：处理过什么类型任务、用了什么方案、效果如何
    ↓
3. 向量相似度预筛选：与现有记忆太相似的候选直接跳过
    ↓
4. LLM 判断每条候选记忆的处理方式：
   CREATE（新建）/ SKIP（跳过）/ MERGE（合并到现有）/ DELETE（删除过时记忆）
    ↓
5. 更新 viking://user/memories/ 和 viking://agent/memories/
   刷新向量索引
```

这个流程解决了"Agent 没有长期记忆"的问题。处理过同类问题、熟悉了用户的偏好之后，Agent 的上下文检索会越来越精准。

设计上值得注意的两点：
- 第 3 步的相似度预筛选避免了记忆库里堆积大量重复内容
- 第 4 步不是简单地"加一条记忆"，而是做了合并/删除的决策，保持记忆库的简洁和一致性

---

## 六、代码库导读

> 这一节是代码导游：入口在哪里、各层之间怎么连、做一件事的时候代码走了哪条路。

### 整体目录结构

OpenViking 是多语言项目，按职责分层很清晰：

```
volcengine/OpenViking/
├── openviking/              # Python 主包（业务逻辑全在这里）
│   ├── __init__.py          # 对外暴露 OpenViking / SyncOpenViking / Session 等
│   ├── client.py            # 重新导出 Sync/Async/HTTP 客户端
│   ├── sync_client.py       # SyncOpenViking（同步封装）
│   ├── async_client.py      # AsyncOpenViking（嵌入模式核心，单例）
│   ├── agfs_manager.py      # 启动/停止 Go agfs-server 子进程
│   ├── service/             # 所有业务服务（写入、检索、会话...）
│   ├── storage/             # 虚拟文件系统 + 向量数据库 + 消息队列
│   ├── session/             # 会话管理 + 记忆提取 + 去重
│   ├── retrieve/            # 层次检索器 + 意图分析
│   ├── models/              # VLM / Embedding 抽象层
│   ├── server/              # HTTP 服务模式（FastAPI）
│   └── prompts/             # 各类 Prompt 模板
│
├── src/          (C++17)    # 高性能 HNSW 向量索引内核
├── third_party/agfs/  (Go)  # agfs-server：底层分布式文件服务
└── crates/ov_cli/  (Rust)   # ov 命令行工具
```

---

### 入口：怎么启动？

#### 嵌入模式（本地开发最常用）

```python
import openviking as ov

client = ov.OpenViking(path="./data")   # 等价于 SyncOpenViking
client.initialize()
```

调用链：

```
SyncOpenViking.initialize()
  └─► AsyncOpenViking.initialize()        # 单例，线程锁保护
        └─► OpenVikingService.initialize() # 核心编排器，顺序执行以下步骤：
              │
              ├─ 1. 读取 ov.conf 配置
              ├─ 2. 初始化 QueueManager（SQLite 队列，两条通道）
              │       ├─ EmbeddingQueue  (max_concurrent=10)
              │       └─ SemanticQueue   (max_concurrent=100)
              ├─ 3. 初始化向量数据库 VikingDBManager
              ├─ 4. 启动 AGFSManager
              │       └─ 写配置 → 启动 agfs-server（Go 二进制）→ 等待健康检查
              ├─ 5. 创建 VikingFS（viking:// 虚拟文件系统）
              ├─ 6. 启动队列 Worker 线程
              ├─ 7. 创建根目录（resources/ memories/ sessions/ 等）
              └─ 8. 注入子服务（FSService / SearchService / ResourceService / SessionService...）
```

#### HTTP 服务模式（多客户端共享）

```bash
# 服务端
python -m openviking.server.bootstrap --host 0.0.0.0 --port 1933
```

```python
# 客户端
client = ov.SyncHTTPClient(url="http://localhost:1933")
```

HTTP 服务启动时走同一套 `OpenVikingService.initialize()`，然后把所有操作包成 FastAPI 路由（14+ 个 router 文件，对应 `/api/v1/resources`、`/api/v1/search`、`/api/v1/sessions` 等）。

---

### 流程一：写入一个文档

```python
client.add_resource(path="https://example.com/doc.pdf", wait=True)
```

这条调用背后发生了什么：

```
ResourceService.add_resource(path, ctx, wait=True)
  │
  ├─ 1. 校验路径（必须属于 resources/ 命名空间）
  │
  ├─ 2. ResourceProcessor.process(path, target_uri, ctx)
  │       ├─ 下载/读取文件内容
  │       └─ VikingFS.write(uri, content)
  │             └─ AGFSClient.write(...)  →  agfs-server  →  本地磁盘 / S3
  │
  ├─ 3. SemanticQueue.enqueue(SemanticMsg)   # 异步，不阻塞
  │       SemanticMsg { uri, context_type="resource", recursive=True }
  │
  └─ 4. [wait=True 时] 等待队列清空
```

SemanticQueue 后台 Worker（SemanticProcessor）接到任务后：

```
SemanticProcessor.process(SemanticMsg)
  │
  ├─ 自底向上遍历目录树
  │
  ├─ 对每个文件：调用 VLM 生成单文件摘要 → 写 .abstract.md
  │
  ├─ 对每个目录：
  │   ├─ 汇总子文件摘要 → 调用 VLM → 写目录 .abstract.md  (~100 token，L0)
  │   └─ 生成结构化概览 → 调用 VLM → 写目录 .overview.md (~2000 token，L1)
  │
  └─ 每生成一个摘要文件，就往 EmbeddingQueue 投一条消息
```

EmbeddingQueue 后台 Worker（EmbeddingProcessor）：

```
EmbeddingProcessor.process(EmbeddingMsg)
  │
  ├─ 判断是 L0 / L1 / L2（从 URI 后缀或 metadata 读取）
  ├─ Embedder.embed(text)  →  dense vector（+ 可选 sparse vector）
  └─ VikingDBManager.upsert(Context)
        └─ VikingVectorIndexBackend.upsert()  →  向量数据库（RocksDB 持久化）
```

一份文档写进去，最终会产生三类落地数据：
- **原始文件**：通过 AGFS → 本地磁盘 / S3
- **语义摘要**：`.abstract.md` + `.overview.md`，也写进 VikingFS
- **向量索引**：摘要和原文的向量，写进 VikingDBManager

---

### 流程二：检索（Agent 搜索上下文）

```python
results = client.find("Python 内存泄漏优化", target_uri="viking://resources/...")
```

```
SearchService.find(query, ctx, target_uri)
  └─► VikingFS.find(query, ctx, target_uri)
        └─► HierarchicalRetriever.retrieve(query, target_uri, ctx)
```

HierarchicalRetriever 是检索的核心，分五个阶段：

```
阶段 1：嵌入查询
  Embedder.embed(query)  →  dense vector

阶段 2：全局 L0 搜索
  VikingDBManager.search_global_roots_in_tenant(
      vector, level=L0, directories=[target_uri]
  )
  → 在所有目录的 .abstract.md 向量里找高分目录（花费最少）

阶段 3：目录递归下钻
  for 每个高分目录:
    VikingDBManager.search_children_in_tenant(vector, parent_uri, level=L1)
    子目录得分 = 0.5 × 父目录得分 + 0.5 × 向量相似度
    → 递归进入高分子目录，直到收敛（连续若干轮结果不变则停止）

阶段 4：叶节点 L2 检索
  VikingDBManager.search_children_in_tenant(vector, dir_uri, level=L2)
  → 只对最终确定的目录，才加载完整原文

阶段 5：可选 Rerank
  [mode="thinking"]  →  调用外部 rerank 服务精排
  [mode="quick"]     →  直接用向量分数

返回：List[MatchedContext]（带 URI、层级、分数、内容）
```

会话感知搜索（`client.search()`）在此基础上多了一步：先用 IntentAnalyzer（LLM）分析当前对话意图，拆成多个带优先级的子查询（resource_query / memory_query / skill_query），再分别走上面的流程。

---

### 流程三：会话结束，提取长期记忆

```python
client.commit_session(session_id)
```

分两个阶段，第一阶段同步、第二阶段后台异步：

```
Phase 1（同步，加文件系统锁）：
  Session.commit_async()
    ├─ 快照当前 messages.jsonl
    ├─ 清空实时消息缓冲
    └─ 写入归档目录：viking://sessions/<user>/<sid>/archives/<N>/messages.jsonl

Phase 2（后台异步）：
  SessionCompressor.extract_long_term_memories(messages)
    │
    ├─ MemoryExtractor（VLM）
    │   ├─ 检测消息语言（正则识别中/日/韩/俄/阿拉伯）
    │   └─ 调用 VLM → 提取候选记忆，分 8 个类别：
    │       用户侧：profile / preferences / entities / events
    │       Agent 侧：cases / patterns / tools / skills
    │
    ├─ 对每条候选记忆，走 MemoryDeduplicator：
    │   ├─ 嵌入候选文本 → 向量
    │   ├─ 在现有记忆里搜索最相似的 Top-5
    │   ├─ [无相似]  →  CREATE：直接写入新记忆文件
    │   └─ [有相似]  →  让 LLM 决策：
    │         MERGE：让 LLM 生成合并版本，覆盖旧记忆
    │         SKIP：与现有记忆重复，丢弃
    │         CREATE：内容有新意，另建文件
    │
    ├─ 写记忆文件到 VikingFS：
    │   viking://memories/<owner_space>/profile.md        # profile 永远只有一个文件，持续 MERGE
    │   viking://memories/<owner_space>/preferences/<uuid>.md
    │   viking://memories/<owner_space>/cases/<uuid>.md
    │   ...
    │
    ├─ 长记忆按段落边界分块 → 全部进 EmbeddingQueue（走向量化流程）
    │
    └─ 重新触发 SemanticProcessor 对 memories/ 目录生成新的 .abstract.md / .overview.md
```

---

### 三个流程的关键数据结构

**`SemanticMsg`**：语义处理队列的消息体

```python
@dataclass
class SemanticMsg:
    id: str           # UUID
    uri: str          # viking://resources/<account>/<path>/
    context_type: str # "resource" | "memory" | "skill" | "session"
    recursive: bool   # 是否自底向上遍历整棵目录树
    account_id: str
    user_id: str
    agent_id: str
```

**`Context`**：向量数据库里的记录单元

```python
@dataclass
class Context:
    uri: str
    context_type: ContextType   # skill | memory | resource
    level: ContextLevel         # abstract(L0) | overview(L1) | detail(L2)
    vector: List[float]
    content: str
    active_count: int           # 访问频率，用于 hotness 加分
    owner_space: str            # 租户隔离键
    account_id: str
```

**`MemoryCategory`**：记忆的 8 个分类

```python
class MemoryCategory(str, Enum):
    PROFILE     = "profile"      # 用户画像，始终单文件
    PREFERENCES = "preferences"  # 偏好
    ENTITIES    = "entities"     # 提到的实体（人/项目/工具）
    EVENTS      = "events"       # 发生的事件
    CASES       = "cases"        # Agent 处理过的案例
    PATTERNS    = "patterns"     # 发现的规律
    TOOLS       = "tools"        # 工具使用经验
    SKILLS      = "skills"       # 技能模板
```

---

### 各组件的分工一览

| 组件 | 语言 | 职责 |
|---|---|---|
| `service/` | Python | 业务入口，编排各流程 |
| `storage/viking_fs.py` | Python | viking:// 虚拟文件系统，URI → AGFS 路径的翻译层 |
| `storage/queuefs/` | Python | 异步队列（Semantic + Embedding），SQLite 持久化 |
| `storage/vectordb/` | Python | 向量数据库门面，管理多租户隔离 |
| `retrieve/hierarchical_retriever.py` | Python | L0→L1→L2 递归检索核心 |
| `session/` | Python | 会话生命周期 + 记忆提取/去重/合并 |
| `models/vlm/` | Python | VLM 抽象（Doubao/OpenAI/LiteLLM 统一接口） |
| `agfs_manager.py` | Python | 管理 Go agfs-server 子进程 |
| `third_party/agfs/` | Go | 底层文件读写（本地磁盘 / S3 / 内存） + SQLite 队列 |
| `src/index/` | C++ | HNSW 向量索引内核 |
| `crates/ov_cli/` | Rust | `ov` 命令行工具 |

AGFS（Go 二进制）承担的职责比较纯粹：它只是一个带 HTTP API 的文件服务器，不含任何业务逻辑。Python 层的 `VikingFS` 负责把 `viking://resources/<account>/xxx` 这样的语义 URI 翻译成 AGFS 能理解的 `/local/<account_id>/resources/xxx` 路径。两者之间是清晰的职责边界。

---

## 七、实测数据

在 **LoCoMo10** 基准（1540 个长对话 Agent 任务）上的测试结果：

| 配置 | 任务完成率 | 输入 Token 数 |
|---|---|---|
| OpenClaw（无 OpenViking） | 35.65% | 24.6M |
| OpenClaw + OpenViking | 52.08% | 4.3M |
| OpenClaw + OpenViking + 本地记忆 | 51.23% | 2.1M |

两个关键数字：
- 任务完成率提升 **+46%**
- Token 消耗降低约 **83%**

这两个数字同时改善并不矛盾——恰恰是因为检索质量更好（只把真正相关的内容放进 Context），所以 LLM 能更精准地完成任务，同时也不需要处理大量无关内容。

---

## 八、设计上的几点思考

**文件系统范式的边界**：文件系统是一个很好的组织模型，但它假设你事先知道该把东西放哪里。对于复杂的跨领域内容，分类本身就是一个难题。OpenViking 通过让 VLM 来决定文件放置路径来缓解这个问题，但这引入了 VLM 分类错误的风险。

**L0/L1/L2 的质量取决于 VLM**：三个层级的生成都依赖 VLM 的摘要能力。如果 VLM 的摘要不准确，整个检索链路的质量都会受影响。尤其是 L0（100 tokens 的极简摘要），信息压缩比非常高，容易丢失关键细节。

**记忆巩固的判断依赖 LLM**：CREATE/SKIP/MERGE/DELETE 的决策是 LLM 做的，这意味着记忆质量本质上受 LLM 能力约束，且不透明。长期运行后记忆库的状态可能难以审计。

**向量数据库的多租户复杂性**：`RequestContext` 携带 `account_id`、`user`、`role` 在每次操作中做隔离，这在分布式部署下会显著增加系统复杂度，需要仔细测试边界情况。

---

## 总结

OpenViking 最值得关注的不是它的某个具体技术点，而是它的**范式选择**：把 Agent 上下文管理从"向量数据库检索问题"重新定义为"文件系统管理问题"。

这个重新定义带来了层次化检索、可调试性、统一的存储模型，以及自然的访问控制语义。L0/L1/L2 分层和目录递归检索是在这个范式上生长出来的自然结论。

当然，这套设计的成熟度还有待验证——项目目前仍是 Alpha 状态。但它提出的问题和解法值得认真思考：在 Agent 时代，"记忆"应该是什么形状的？
