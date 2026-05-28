---
date: 2026-03-28
title: "OpenViking: Redefining Agent Memory and Retrieval with a Filesystem"
description: A deep dive into the core design of ByteDance's OpenViking — why use the filesystem paradigm to manage agent context, and how L0/L1/L2 hierarchical retrieval simultaneously improves accuracy and cuts token consumption.
tags: [agent, rag, memory, retrieval, context-management, openviking]
---

# OpenViking: Redefining Agent Memory and Retrieval with a Filesystem

> Source: [github.com/volcengine/OpenViking](https://github.com/volcengine/OpenViking) (ByteDance Volcengine, open-sourced January 2026, Apache 2.0)

---

## 1. The Agent Memory Problem

Before we get into OpenViking, ask this: **what kind of "memory" does a continuously running AI agent actually need?**

It's not a trivial question. Mainstream RAG systems (vectorize documents, run similarity search at query time) are fine for simple Q&A, but drop them into a real agent workflow and several core tensions surface fast:

**1. Context comes from everywhere**

An agent's context comes from multiple places — user history (memory), tool-produced documents (resources), and acquired skills (prompt templates and tool-calling patterns). In traditional architectures these live scattered across vector DBs, relational DBs, and prompt folders. For the agent to "remember" something, it has to query several systems separately, and the logic ends up fragmented.

**2. Token consumption explodes**

The crude fix is "stuff all retrieved documents into context." But once the docs get long or numerous, token counts balloon. A long-running coding agent working on a complex task can easily blow past 100K tokens between relevant code files, conversation history, and technical docs — and that's not just expensive, it also degrades quality because the attention mechanism gets diluted.

**3. Retrieval quality is poor**

Flat vector search lacks hierarchy. Search for "how to optimize this code" and you may get back 20 scattered snippets instead of "first figure out which module the problem is in, then find the core logic of that module." Human experts solve problems top-down; RAG does bottom-up brute-force matching.

**4. Memory doesn't update itself**

After an agent finishes a task, what has it learned? Nothing. Every conversation starts from scratch — previously solved problems, user preferences, lessons learned — all gone.

OpenViking's central claim: **the root cause of all of this is the wrong storage paradigm.**

---

## 2. The Core Idea: A Filesystem Paradigm

OpenViking's solution is surprisingly intuitive — it organizes all agent context, whether memory, documents, or skills, into a single **virtual filesystem**, accessed via the `viking://` URI scheme.

```
viking://
├── resources/              # Account-level shared documents/data
├── user/{space}/           # User-private memory (access-isolated)
│   ├── profile/            # User preferences and background
│   └── history/            # Conversation history summaries
├── agent/{space}/          # Agent-specific space
│   ├── skills/             # Skills: prompt templates, tool-calling patterns
│   └── memories/           # Agent's accumulated experience
└── session/{id}/           # Session-level transient context
    ├── context/            # Relevant material for the current task
    └── history/            # Current conversation log
```

The agent interacts with this filesystem exactly like a Unix shell: `ls`, `find`, `grep`, `tree`, and there's even an `ov` CLI tool.

A few deeper implications of this design choice:

**Unified mental model**: memory, resources, and skills are no longer three different systems — they're all "files" in different "directories." The agent just needs to know how to "look up files"; it doesn't need to learn three sets of APIs.

**Built-in hierarchy**: a filesystem is tree-shaped by nature. That gives hierarchical retrieval a physical basis — "find the right directory first, then the right file" — instead of brute-forcing across every file.

**Isolation and sharing coexist**: `resources/` is globally shared, `user/` is user-private, `session/` is transient. Access control maps naturally to directory permissions.

**Debuggability**: filesystem paths are human-readable. You can see that the agent read from `viking://agent/coding/skills/refactoring.md`, instead of "fetched 5 chunks from the vector DB."

---

## 3. L0 / L1 / L2 Hierarchical Retrieval: Defusing the Token Bomb

The filesystem solves "where to put things" but not "how much to fetch." OpenViking's second core design is **semantic hierarchical storage**.

Every file written into OpenViking is automatically given three semantic levels by the `SemanticProcessor`:

| Level | Token budget | How it's generated | Purpose |
|---|---|---|---|
| **L0 abstract** | ~100 tokens | VLM-generated minimal summary | Vector index + directory-level filtering |
| **L1 overview** | ~2,000 tokens | Hierarchical structured summary | Reranking + navigation decisions |
| **L2 detail** | Unlimited | Raw content | Loaded on demand only |

Retrieval proceeds in layers:

```
Query: "Optimize memory leaks in Python code"
    ↓
1. Semantic expansion: generate 3–5 sub-queries
   ["memory leak detection", "Python gc module", "weakref", ...]
    ↓
2. Vector search over all files' L0 abstracts (cheap)
   → 3 high-scoring directories hit
    ↓
3. Recursive search inside those directories
   → individual files' L0 hit, rerank using L1
    ↓
4. Only load L2 detail for the files we actually need
```

The key insight: **most retrieval tasks can make their navigation decisions at L0; L2 is only paid for when detail genuinely matters.**

Compared to traditional RAG: classical RAG chops documents into fixed-size chunks at index time and vectorizes them indiscriminately. Two problems follow. First, semantics get cut arbitrarily — a code comment and its function get split apart. Second, at query time the full chunk is pushed into context whether it's actually needed or not.

The L0/L1/L2 split makes token cost on-demand — only files you've actually committed to reading pay the cost of their full content.

---

## 4. Recursive Directory Retrieval: Thinking Like a Human Expert

OpenViking's retrieval algorithm has one elegant trick: **scores propagate from child nodes up to parent directories.**

Concretely, when a file's L0 abstract scores highly in vector search, its parent directory inherits a derived score:

```
directory_score = 0.5 × parent_directory_score + 0.5 × embedding_similarity
```

This formula lets the retriever make a judgment traditional RAG simply can't: **"this directory contains many highly relevant files, so the directory itself is relevant and worth searching deeper into."**

That mirrors how a human expert finds relevant code in a large codebase: first decide "this looks like an infrastructure bug, head to `infra/`," then "specifically a networking issue, drill into `infra/networking/`," and only then open individual files.

The recursion's depth and breadth are configurable. The output isn't a pile of scattered document fragments — it's a **hierarchical retrieval tree with path information attached.**

---

## 5. Self-Evolving Memory After Each Session

After every conversation ends, OpenViking automatically runs a "memory consolidation" pass:

```
Conversation ends
    ↓
1. Archive full conversation + used context to session/{id}/history/
    ↓
2. VLM extracts structured memory fragments from the conversation
   - User-side: preferences, work background, common patterns
   - Agent-side: task types handled, solutions used, outcomes
    ↓
3. Vector similarity pre-filter: candidates too similar to existing memories are dropped
    ↓
4. LLM decides what to do with each candidate:
   CREATE / SKIP / MERGE (into an existing memory) / DELETE (stale memory)
    ↓
5. Update viking://user/memories/ and viking://agent/memories/
   Refresh vector indexes
```

This is the answer to "agents don't have long-term memory." Once the agent has handled similar problems before and learned a user's preferences, context retrieval gets sharper and sharper.

Two design details worth noting:
- Step 3's similarity pre-filter prevents the memory store from accumulating piles of duplicates.
- Step 4 isn't a naive append — it makes merge/delete decisions, keeping the memory store clean and consistent.

---

## 6. A Tour of the Codebase

> This section is a guided tour: where the entry points are, how the layers connect, and which path code actually takes when doing X.

### Overall directory layout

OpenViking is a multi-language project with clean separation of concerns:

```
volcengine/OpenViking/
├── openviking/              # Python main package (all business logic here)
│   ├── __init__.py          # Exports OpenViking / SyncOpenViking / Session, etc.
│   ├── client.py            # Re-exports Sync/Async/HTTP clients
│   ├── sync_client.py       # SyncOpenViking (sync wrapper)
│   ├── async_client.py      # AsyncOpenViking (embedded-mode core, singleton)
│   ├── agfs_manager.py      # Starts/stops the Go agfs-server subprocess
│   ├── service/             # All business services (ingest, retrieve, session...)
│   ├── storage/             # Virtual filesystem + vector DB + message queue
│   ├── session/             # Session management + memory extraction + dedup
│   ├── retrieve/            # Hierarchical retriever + intent analysis
│   ├── models/              # VLM / Embedding abstraction layer
│   ├── server/              # HTTP server mode (FastAPI)
│   └── prompts/             # Various prompt templates
│
├── src/          (C++17)    # High-perf HNSW vector index core
├── third_party/agfs/  (Go)  # agfs-server: low-level distributed file service
└── crates/ov_cli/  (Rust)   # The ov CLI tool
```

---

### Entry point: how do you start it?

#### Embedded mode (most common in local development)

```python
import openviking as ov

client = ov.OpenViking(path="./data")   # equivalent to SyncOpenViking
client.initialize()
```

Call chain:

```
SyncOpenViking.initialize()
  └─► AsyncOpenViking.initialize()        # singleton, guarded by thread lock
        └─► OpenVikingService.initialize() # core orchestrator, runs the following in order:
              │
              ├─ 1. Read ov.conf
              ├─ 2. Initialize QueueManager (SQLite queue, two channels)
              │       ├─ EmbeddingQueue  (max_concurrent=10)
              │       └─ SemanticQueue   (max_concurrent=100)
              ├─ 3. Initialize the vector DB (VikingDBManager)
              ├─ 4. Start AGFSManager
              │       └─ Write config → launch agfs-server (Go binary) → wait for health check
              ├─ 5. Create VikingFS (viking:// virtual filesystem)
              ├─ 6. Start queue worker threads
              ├─ 7. Create root directories (resources/, memories/, sessions/, ...)
              └─ 8. Inject sub-services (FSService / SearchService / ResourceService / SessionService...)
```

#### HTTP server mode (multiple clients sharing)

```bash
# Server
python -m openviking.server.bootstrap --host 0.0.0.0 --port 1933
```

```python
# Client
client = ov.SyncHTTPClient(url="http://localhost:1933")
```

The HTTP server boots through the same `OpenVikingService.initialize()`, then wraps all operations as FastAPI routes (14+ router files covering `/api/v1/resources`, `/api/v1/search`, `/api/v1/sessions`, etc.).

---

### Flow 1: writing a document

```python
client.add_resource(path="https://example.com/doc.pdf", wait=True)
```

What actually happens behind this one call:

```
ResourceService.add_resource(path, ctx, wait=True)
  │
  ├─ 1. Validate path (must live under the resources/ namespace)
  │
  ├─ 2. ResourceProcessor.process(path, target_uri, ctx)
  │       ├─ Download/read file contents
  │       └─ VikingFS.write(uri, content)
  │             └─ AGFSClient.write(...)  →  agfs-server  →  local disk / S3
  │
  ├─ 3. SemanticQueue.enqueue(SemanticMsg)   # async, non-blocking
  │       SemanticMsg { uri, context_type="resource", recursive=True }
  │
  └─ 4. [if wait=True] block until the queue drains
```

When the SemanticQueue's background worker (SemanticProcessor) picks up the job:

```
SemanticProcessor.process(SemanticMsg)
  │
  ├─ Traverse the directory tree bottom-up
  │
  ├─ For each file: call VLM to generate per-file summary → write .abstract.md
  │
  ├─ For each directory:
  │   ├─ Aggregate child file summaries → call VLM → write directory .abstract.md  (~100 tokens, L0)
  │   └─ Generate structured overview → call VLM → write directory .overview.md (~2000 tokens, L1)
  │
  └─ Every summary file generated triggers a message on the EmbeddingQueue
```

EmbeddingQueue background worker (EmbeddingProcessor):

```
EmbeddingProcessor.process(EmbeddingMsg)
  │
  ├─ Determine L0 / L1 / L2 (from URI suffix or metadata)
  ├─ Embedder.embed(text)  →  dense vector (+ optional sparse vector)
  └─ VikingDBManager.upsert(Context)
        └─ VikingVectorIndexBackend.upsert()  →  vector DB (RocksDB-persisted)
```

One document ingested produces three categories of persisted data:
- **Raw file**: via AGFS → local disk / S3
- **Semantic summaries**: `.abstract.md` + `.overview.md`, also written into VikingFS
- **Vector indexes**: embeddings of summaries and raw content, written into VikingDBManager

---

### Flow 2: retrieval (the agent searching for context)

```python
results = client.find("Python memory leak optimization", target_uri="viking://resources/...")
```

```
SearchService.find(query, ctx, target_uri)
  └─► VikingFS.find(query, ctx, target_uri)
        └─► HierarchicalRetriever.retrieve(query, target_uri, ctx)
```

HierarchicalRetriever is the heart of retrieval, in five stages:

```
Stage 1: embed the query
  Embedder.embed(query)  →  dense vector

Stage 2: global L0 search
  VikingDBManager.search_global_roots_in_tenant(
      vector, level=L0, directories=[target_uri]
  )
  → Search all directories' .abstract.md vectors for high-scoring directories (cheapest stage)

Stage 3: recursive directory drill-down
  for each high-scoring directory:
    VikingDBManager.search_children_in_tenant(vector, parent_uri, level=L1)
    child_score = 0.5 × parent_score + 0.5 × embedding_similarity
    → Recurse into high-scoring subdirectories until convergence (stops once results stop changing for several rounds)

Stage 4: leaf-node L2 retrieval
  VikingDBManager.search_children_in_tenant(vector, dir_uri, level=L2)
  → Only the finally-selected directories pay the cost of loading full content

Stage 5: optional rerank
  [mode="thinking"]  →  call external rerank service for fine-grained ranking
  [mode="quick"]     →  use vector scores directly

Return: List[MatchedContext] (with URI, level, score, content)
```

Session-aware search (`client.search()`) adds one more step on top: first use IntentAnalyzer (LLM) to analyze the current conversation's intent and break it into multiple prioritized sub-queries (resource_query / memory_query / skill_query), then run each through the pipeline above.

---

### Flow 3: end of session, extracting long-term memory

```python
client.commit_session(session_id)
```

Two phases — phase 1 synchronous, phase 2 background async:

```
Phase 1 (sync, with filesystem lock):
  Session.commit_async()
    ├─ Snapshot the current messages.jsonl
    ├─ Clear the live message buffer
    └─ Write to archive directory: viking://sessions/<user>/<sid>/archives/<N>/messages.jsonl

Phase 2 (background async):
  SessionCompressor.extract_long_term_memories(messages)
    │
    ├─ MemoryExtractor (VLM)
    │   ├─ Detect message language (regex for Chinese/Japanese/Korean/Russian/Arabic)
    │   └─ Call VLM → extract candidate memories across 8 categories:
    │       User-side:  profile / preferences / entities / events
    │       Agent-side: cases / patterns / tools / skills
    │
    ├─ For each candidate, run MemoryDeduplicator:
    │   ├─ Embed candidate text → vector
    │   ├─ Search the existing memory store for top-5 most similar
    │   ├─ [no match]   →  CREATE: write a new memory file directly
    │   └─ [has match]  →  let the LLM decide:
    │         MERGE:  LLM generates a merged version, overwriting the old memory
    │         SKIP:   duplicate of existing memory, discard
    │         CREATE: content is genuinely new, write a separate file
    │
    ├─ Write memory files to VikingFS:
    │   viking://memories/<owner_space>/profile.md        # profile is always a single file, perpetually MERGEd
    │   viking://memories/<owner_space>/preferences/<uuid>.md
    │   viking://memories/<owner_space>/cases/<uuid>.md
    │   ...
    │
    ├─ Long memories are chunked at paragraph boundaries → all enter the EmbeddingQueue (vectorization pipeline)
    │
    └─ Re-trigger SemanticProcessor on the memories/ directory to regenerate .abstract.md / .overview.md
```

---

### Key data structures across the three flows

**`SemanticMsg`**: payload for the semantic-processing queue

```python
@dataclass
class SemanticMsg:
    id: str           # UUID
    uri: str          # viking://resources/<account>/<path>/
    context_type: str # "resource" | "memory" | "skill" | "session"
    recursive: bool   # whether to traverse the directory tree bottom-up
    account_id: str
    user_id: str
    agent_id: str
```

**`Context`**: the record unit in the vector DB

```python
@dataclass
class Context:
    uri: str
    context_type: ContextType   # skill | memory | resource
    level: ContextLevel         # abstract(L0) | overview(L1) | detail(L2)
    vector: List[float]
    content: str
    active_count: int           # access frequency, used for a hotness boost
    owner_space: str            # tenant isolation key
    account_id: str
```

**`MemoryCategory`**: the 8 memory categories

```python
class MemoryCategory(str, Enum):
    PROFILE     = "profile"      # user profile, always single file
    PREFERENCES = "preferences"  # preferences
    ENTITIES    = "entities"     # mentioned entities (people/projects/tools)
    EVENTS      = "events"       # events that happened
    CASES       = "cases"        # cases the agent has handled
    PATTERNS    = "patterns"     # patterns discovered
    TOOLS       = "tools"        # tool-usage experience
    SKILLS      = "skills"       # skill templates
```

---

### Component responsibilities at a glance

| Component | Language | Responsibility |
|---|---|---|
| `service/` | Python | Business entry point, orchestrates the flows |
| `storage/viking_fs.py` | Python | viking:// virtual filesystem, URI → AGFS path translation layer |
| `storage/queuefs/` | Python | Async queues (Semantic + Embedding), SQLite-persisted |
| `storage/vectordb/` | Python | Vector DB facade, manages multi-tenant isolation |
| `retrieve/hierarchical_retriever.py` | Python | L0→L1→L2 recursive retrieval core |
| `session/` | Python | Session lifecycle + memory extraction/dedup/merge |
| `models/vlm/` | Python | VLM abstraction (unified interface over Doubao/OpenAI/LiteLLM) |
| `agfs_manager.py` | Python | Manages the Go agfs-server subprocess |
| `third_party/agfs/` | Go | Low-level file I/O (local disk / S3 / memory) + SQLite queue |
| `src/index/` | C++ | HNSW vector index core |
| `crates/ov_cli/` | Rust | The `ov` CLI tool |

AGFS (the Go binary) has a deliberately narrow responsibility: it's just a file server with an HTTP API, no business logic. The Python `VikingFS` layer translates semantic URIs like `viking://resources/<account>/xxx` into AGFS-understandable paths like `/local/<account_id>/resources/xxx`. The boundary between them is clean.

---

## 7. Benchmark Numbers

Results on the **LoCoMo10** benchmark (1540 long-conversation agent tasks):

| Configuration | Task completion rate | Input tokens |
|---|---|---|
| OpenClaw (no OpenViking) | 35.65% | 24.6M |
| OpenClaw + OpenViking | 52.08% | 4.3M |
| OpenClaw + OpenViking + local memory | 51.23% | 2.1M |

Two numbers worth noting:
- Task completion up **+46%**
- Token consumption down roughly **83%**

These two improvements aren't contradictory — they're causally linked. Better retrieval quality (only the actually-relevant content lands in context) means the LLM can complete tasks more precisely, and it doesn't have to wade through irrelevant material.

---

## 8. Some Design Reflections

**The limits of the filesystem paradigm**: a filesystem is a great organizing model, but it assumes you know up front where things should go. For complex cross-domain content, categorization is itself the hard problem. OpenViking mitigates this by letting the VLM decide placement, but that introduces VLM misclassification as a new risk.

**L0/L1/L2 quality is bottlenecked by the VLM**: all three levels are generated by the VLM. If its summaries are off, the entire retrieval chain degrades. L0 especially (a 100-token minimal summary) has a very high compression ratio and easily loses critical detail.

**Memory consolidation decisions ride on the LLM**: CREATE/SKIP/MERGE/DELETE is decided by an LLM, which means memory quality is fundamentally bounded by LLM capability and the decisions are opaque. After long-term operation, the state of the memory store may be hard to audit.

**Multi-tenant complexity in the vector DB**: `RequestContext` carries `account_id`, `user`, and `role` through every operation for isolation. In a distributed deployment this adds real complexity and demands careful edge-case testing.

---

## Wrap-up

The most interesting thing about OpenViking isn't any single technical point — it's the **paradigm choice**: redefining agent context management from "a vector DB retrieval problem" into "a filesystem management problem."

That redefinition is what enables hierarchical retrieval, debuggability, a unified storage model, and natural access-control semantics. L0/L1/L2 layering and recursive directory retrieval are natural consequences that grow out of the paradigm.

How well this design scales remains to be seen — the project is still Alpha. But the question it poses, and the answer it proposes, are worth taking seriously: in the age of agents, what shape should "memory" take?
