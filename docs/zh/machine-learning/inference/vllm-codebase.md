---
date: 2026-03-28
title: "vLLM 源码拆解：从一个请求到 Token 输出的完整链路"
description: 深度剖析 vLLM 代码库架构，追踪一个推理请求从 API 入口到 Token 输出的完整调用链，理解 Scheduler、Worker、KV Cache Manager 等核心组件的实现
tags: [vllm, inference, source-code, scheduler, kv-cache, serving]
---

# vLLM 源码拆解：从一个请求到 Token 输出的完整链路

> 本文基于 vLLM v1 架构（截至 2026 年 3 月 main 分支）。源码：[github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)。建议先阅读 [第一篇：vLLM 与 PagedAttention](./vllm-pagedattention)，了解 PagedAttention 的算法原理。

第一篇文章我们从论文角度理解了 PagedAttention **做了什么**。这篇文章我们深入源码，看看 vLLM **是怎么做的**——一个推理请求从进入 API 到返回生成的 Token，究竟经历了哪些组件、哪些步骤。

## 代码库全景

先看 vLLM 仓库的顶层目录结构：

```
vllm/
├── vllm/                  # 核心 Python 库
│   ├── v1/                # ⭐ v1 新架构（当前主力）
│   │   ├── engine/        #    引擎层：AsyncLLM, EngineCore, LLMEngine
│   │   ├── core/          #    调度层：Scheduler, KVCacheManager, BlockPool
│   │   ├── worker/        #    执行层：Worker, GPUModelRunner
│   │   └── executor/      #    分发层：UniprocExecutor, MultiprocExecutor
│   ├── engine/            # 旧架构引擎（正在被淘汰）
│   ├── entrypoints/       # API 入口：OpenAI 兼容服务器、CLI、离线批量
│   ├── config/            # 配置模块（27 个配置类）
│   ├── model_executor/    # 模型加载、算子、各模型实现
│   ├── distributed/       # 分布式通信原语
│   ├── multimodal/        # 多模态输入（图像、音频、视频）
│   └── sampling_params.py # 采样参数
├── csrc/                  # C++/CUDA 自定义 kernel
├── benchmarks/            # 性能基准测试
├── tests/                 # 测试套件
└── examples/              # 使用示例
```

::: info v1 架构 vs 旧架构
vLLM 代码库中同时存在两套架构：`vllm/engine/` 下的旧架构（单体设计）和 `vllm/v1/` 下的新 v1 架构（多进程设计）。v1 是当前的开发重心，本文只分析 v1 架构。
:::

## 架构总览：三个进程

v1 架构最显著的特征是 **多进程分离**。一个 vLLM serving 实例由三类进程组成：

```
┌─────────────────────────────────────────────────────────┐
│                    API Server 进程                       │
│                                                         │
│  FastAPI (OpenAI-compatible API)                        │
│       │                                                 │
│       ▼                                                 │
│  AsyncLLM                                               │
│   ├─ InputProcessor   （请求 → EngineCoreRequest）       │
│   ├─ OutputProcessor  （EngineCoreOutputs → 流式响应）   │
│   └─ Detokenizer      （增量 detokenize）               │
│       │                                                 │
│       │ ZMQ IPC                                         │
├───────┼─────────────────────────────────────────────────┤
│       ▼             EngineCore 进程                      │
│  EngineCoreProc                                         │
│   ├─ Scheduler       （决定本轮执行哪些请求）             │
│   ├─ KVCacheManager  （管理 KV block 的分配与回收）       │
│   └─ Executor        （将计算任务分发给 Worker）          │
│       │                                                 │
│       │ SharedMemory MessageQueue                       │
├───────┼─────────────────────────────────────────────────┤
│       ▼             Worker 进程（每 GPU 一个）            │
│  Worker                                                 │
│   └─ GPUModelRunner  （准备输入张量 + 执行模型前向传播）   │
└─────────────────────────────────────────────────────────┘
```

**为什么要多进程？** Python 的 GIL（全局解释器锁）会让多线程在 CPU 密集型任务上变成单线程。将 API 处理、调度逻辑、GPU 计算分到不同进程，避免了 GIL 竞争，实现了真正的并行。

进程间通信使用两套机制：
- **ZMQ IPC**：API Server 进程 ↔ EngineCore 进程
- **SharedMemory MessageQueue**：EngineCore 进程 ↔ Worker 进程（基于共享内存，零拷贝，超低延迟）

## 请求的完整生命周期

让我们追踪一个请求从进入到返回的完整路径。

### 第一站：API 入口

用户发送一个 OpenAI 格式的请求：

```bash
curl http://localhost:8000/v1/completions \
  -d '{"model": "Qwen/Qwen3-8B", "prompt": "Hello", "max_tokens": 100}'
```

请求首先到达 FastAPI 路由层（`vllm/entrypoints/openai/api_server.py`），然后交给 `AsyncLLM`：

```
HTTP Request
  → FastAPI route handler
    → AsyncLLM.generate()
```

`AsyncLLM`（`vllm/v1/engine/async_llm.py`）是 API Server 进程中的核心类。它做两件事：

1. **InputProcessor** 将原始请求转换为 `EngineCoreRequest`（包含 token IDs、采样参数等）
2. 通过 ZMQ 将 `EngineCoreRequest` 发送给 EngineCore 进程

### 第二站：EngineCore 主循环

EngineCore（`vllm/v1/engine/core.py`）运行在独立进程中，执行一个永不停歇的忙循环（busy loop）：

```python
# vllm/v1/engine/core.py — 简化版主循环
def run_busy_loop(self):
    while True:
        # 1. 读取新请求
        self._process_input_queue()

        # 2. 执行一步推理
        outputs = self._process_engine_step()

        # 3. 将结果发回 API Server 进程
        self._process_output_queue(outputs)
```

每一次循环就是一个 **iteration**（迭代）。`_process_engine_step()` 是核心，展开来看：

```python
def _process_engine_step(self):
    # Step 1: Scheduler 决定本轮执行什么
    scheduler_output = self.scheduler.schedule()

    # Step 2: Executor 将任务分发给 Worker 执行
    model_output = self.executor.execute_model(scheduler_output)

    # Step 3: Scheduler 处理模型输出，更新状态
    engine_core_outputs = self.scheduler.update_from_output(model_output)

    return engine_core_outputs
```

这三步就是 vLLM 推理引擎的心跳。下面我们逐一拆解。

### 第三站：Scheduler——每一轮的"决策者"

Scheduler（`vllm/v1/core/sched/scheduler.py`）是 vLLM 的大脑。它维护两个队列：

- **`running`**：当前正在执行的请求列表
- **`waiting`**：等待被调度的请求队列

每次 `schedule()` 调用分为 **两个阶段**：

#### Phase 1：调度正在运行的请求

对 `running` 列表中的每个请求：

1. 计算本轮需要处理的 token 数：`num_tokens = num_tokens_with_spec - num_computed_tokens`
2. 通过 `KVCacheManager.allocate_slots()` 为新 token 分配 KV cache block
3. 如果分配失败（显存不足）→ **抢占（preempt）** 优先级最低的 running 请求，释放其 block，重试

```python
# 简化版 Phase 1 逻辑
for request in self.running:
    num_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_tokens)

    if new_blocks is None:
        # 显存不足，抢占最低优先级的请求
        victim = self.running.pop()  # 最后一个 = 最低优先级
        self.kv_cache_manager.free(victim)
        victim.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(victim)  # 放回等待队列头部
```

#### Phase 2：调度等待中的请求

只有在 Phase 1 没有发生抢占时才执行：

1. 按策略（FCFS 或 Priority）从 `waiting` 队列取出请求
2. 查询前缀缓存命中：`KVCacheManager.get_computed_blocks()`
3. 计算需要调度的新 token 数，检查 token budget
4. 分配 KV cache，将请求移入 `running`

关键约束条件：
- **`max_num_batched_tokens`**：每个 iteration 的 token 上限
- **`max_num_seqs`**：每个 iteration 的最大并发序列数
- **`max_model_len`**：模型支持的最大序列长度

<HtmlVisualization
  src="/machine-learning/inference/visualizations/scheduler-two-phase.html"
  height="660px"
  title="Scheduler 两阶段调度过程"
/>

#### 构建 SchedulerOutput

两个阶段完成后，Scheduler 构建 `SchedulerOutput`，包含：

| 字段 | 说明 |
|---|---|
| `scheduled_new_reqs` | 首次被调度的请求（需传输完整数据：token IDs、采样参数等） |
| `scheduled_cached_reqs` | 已在运行的请求（只传增量更新） |
| `num_scheduled_tokens` | 每个请求本轮要处理的 token 数 |
| `total_num_scheduled_tokens` | 所有请求的 token 总数 |

::: tip Continuous Batching
与传统的静态 batching（一个 batch 必须等所有请求都完成才能处理下一个 batch）不同，vLLM 的 Scheduler **每个 iteration 都重新决策**。请求可以在任意 iteration 被加入或移出 batch。这就是 Continuous Batching（连续批处理），是 vLLM 高吞吐的关键之一。
:::

<HtmlVisualization
  src="/machine-learning/inference/visualizations/continuous-batching.html"
  height="620px"
  title="Static Batching vs Continuous Batching"
/>

### 第四站：Executor——分发计算任务

Executor 负责将 `SchedulerOutput` 分发给 GPU Worker 执行。根据部署场景，有不同的实现：

| Executor | 适用场景 |
|---|---|
| `UniprocExecutor` | 单进程单 GPU |
| `MultiprocExecutor` | 单机多 GPU（多进程） |
| `RayExecutor` | 多机多 GPU（通过 Ray 分布式框架） |

`MultiprocExecutor` 使用 **SharedMemory MessageQueue** 广播 `SchedulerOutput` 给所有 Worker，然后等待 Worker 返回结果。IPC 基于共享内存实现，避免了数据拷贝的开销。

### 第五站：Worker + GPUModelRunner——真正的推理执行

Worker（`vllm/v1/worker/gpu_worker.py`）是每个 GPU 上的执行者。它把工作委托给 `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`），后者负责：

1. **准备输入张量**：从 `SchedulerOutput` 构建模型需要的 input_ids、position_ids、attention metadata 等
2. **执行模型前向传播**：调用模型的 `forward()` 方法
3. **采样**：根据 `SamplingParams` 从 logits 中采样出下一个 token
4. **异步输出传输**：使用独立的 CUDA stream 将结果从 GPU 传回 CPU

```python
# 简化版 Worker.execute_model()
def execute_model(self, scheduler_output):
    # 1. 准备输入
    inputs = self.model_runner.prepare_inputs(scheduler_output)

    # 2. 模型前向传播
    hidden_states = self.model.forward(**inputs)

    # 3. 采样
    sampled_tokens = self.model_runner.sample(hidden_states)

    return ModelRunnerOutput(sampled_tokens=sampled_tokens)
```

`GPUModelRunner` 还管理一个关键优化——**CUDA Graph**。它会在预热阶段捕获不同 batch size 的计算图，推理时直接重放，避免每次 iteration 重新构建 CUDA kernel 调用序列。

### 第六站：结果回传

```
ModelRunnerOutput (Worker)
  → SharedMemory MQ → Executor
    → Scheduler.update_from_output()
      → 检查停止条件（EOS、stop strings、max_tokens）
      → 更新 request 状态
      → 释放已完成请求的 KV cache
    → EngineCoreOutputs
      → ZMQ → AsyncLLM
        → OutputProcessor.process_outputs()
          → Detokenizer（增量反 tokenize）
          → RequestOutput
            → 流式返回给客户端
```

`Scheduler.update_from_output()` 检查多种停止条件：
- 生成了 EOS token
- 命中了 `stop` 字符串
- 达到了 `max_tokens` 限制
- 序列长度达到 `max_model_len`

满足任一条件，请求被标记为完成，其 KV cache block 被释放回 `BlockPool`。

## PagedAttention 在代码中的实现

第一篇文章讲了 PagedAttention 的算法原理。那么在代码中，"按 block 读取 K/V"这件事到底是怎么做的？

### 两层实现：CUDA Kernel 和 FlashAttention

vLLM 实际上有 **两套** PagedAttention 的实现：

| 实现 | 位置 | 用途 |
|---|---|---|
| **自定义 CUDA Kernel** | `csrc/attention/attention_kernels.cuh` | 旧架构（v0）使用 |
| **FlashAttention / FlashInfer** | `vllm/v1/attention/backends/` | v1 架构使用（当前主力） |

在 v1 架构中，vLLM 不再自己写 attention 的 CUDA kernel，而是将 block_table 直接传给 FlashAttention/FlashInfer，让它们处理分页读取。但理解自定义 kernel 的实现对理解 PagedAttention 原理更有帮助，所以我们两个都看。

### 自定义 CUDA Kernel：手动按 block 查表

核心代码在 `csrc/attention/attention_kernels.cuh`，kernel 签名（简化版）：

```cpp
__device__ void paged_attention_kernel(
    scalar_t* __restrict__ out,           // 输出
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq] ← 页表！
    const int* __restrict__ seq_lens,     // [num_seqs]
    ...)
```

**关键操作——查页表**：

```cpp
// 拿到当前序列的页表行
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// 遍历每个逻辑 block
for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
    // 查表：逻辑 block → 物理 block
    const int64_t physical_block_number = block_table[block_idx];

    // 用物理 block 号计算 K 的内存地址
    const cache_t* k_ptr = k_cache
        + physical_block_number * kv_block_stride   // 跳到物理 block
        + kv_head_idx * kv_head_stride;              // 跳到对应 head

    // 计算 qk = scale * dot(q, k)
    // ...
}
```

这就是第一篇文章中"逻辑 block → 物理 block"映射的实际代码。`block_table[block_idx]` 就是那个"页表查询"——给定逻辑 block 编号，返回物理 block 编号，然后用物理编号去 GPU 显存中读取实际的 K/V 数据。

**K 和 V 的显存布局**不同，是为了优化访问模式：

```
K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    ↑ 交错布局，让线程组做 dot(q, k) 时能 coalesced memory access

V cache: [num_blocks, num_kv_heads, head_size, block_size]
    ↑ 转置布局，方便做 weighted sum: output += attention_score * v
```

**Kernel 的完整计算流程：**

1. 加载当前序列的 query 向量到共享内存
2. 遍历所有逻辑 block：通过 `block_table[block_idx]` 查到物理 block 号
3. 从物理 block 加载 K 向量，计算 `qk = scale × dot(q, k)`
4. 对所有 qk scores 做 softmax（先取 max，再 exp 归一化）
5. 再次遍历所有 block：加载 V 向量，做 `output += softmax_score × v`
6. 跨 warp 规约，写出最终结果

::: details V1 Kernel vs V2 Kernel
自定义 kernel 有两个版本：
- **V1**：每个线程块处理一个 head 的**完整**序列。适合短序列。
- **V2**：将序列 **分区（partition）** 到多个线程块并行处理，再用一个 reduce kernel 合并结果（利用 online softmax 技巧合并独立计算的 softmax 分区）。适合长序列。

选择逻辑大致为：序列长度 > 阈值时用 V2，否则用 V1。
:::

### v1 架构：FlashAttention 的分页支持

在 v1 架构中，vLLM 不再直接调用上述 CUDA kernel，而是使用 FlashAttention（或 FlashInfer）原生的分页 KV cache 支持。核心代码在 `vllm/v1/attention/backends/flash_attn.py`：

```python
# vllm/v1/attention/backends/flash_attn.py — 简化版
flash_attn_varlen_func(
    q=query,
    k=key_cache,              # 整个 KV cache 池
    v=value_cache,
    cu_seqlens_q=cu_seqlens_q, # 每个序列的 query 起始位置
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,       # 每个序列已有的 KV 长度
    max_seqlen_k=max_seqlen_k,
    block_table=block_table,   # ← 页表直接传给 FlashAttention！
    softmax_scale=self.scale,
    causal=True,
)
```

关键点：**同一个 `block_table` 张量**被直接传给了 FlashAttention。FlashAttention 内部使用和自定义 kernel 相同的逻辑——通过页表查找物理 block 号，从非连续的物理地址读取 K/V。只是这个实现被封装在了 FlashAttention 库内部。

### Block Table 的数据流

Block table 从 Scheduler 到 Kernel 的完整流动路径：

```
Scheduler 分配逻辑 block → 物理 block 映射
    │
    ▼ (SchedulerOutput 中携带 block_ids)
GPUModelRunner._update_states()
    │  写入 BlockTable 对象的 CPU buffer
    ▼
BlockTable.commit_block_table()
    │  CPU → GPU 拷贝（DMA transfer）
    ▼
block_table_tensor (GPU 上的 int32 张量)
    │  shape: [num_reqs, max_blocks_per_req]
    ▼
CommonAttentionMetadata
    │  包装为 FlashAttentionMetadata.block_table
    ▼
flash_attn_varlen_func(block_table=...)
    │  FlashAttention 内部按 block 查表读取 K/V
    ▼
Attention 输出
```

### Slot Mapping：KV Cache 的写入

读取 K/V 时用 `block_table`，但 **写入** 新的 K/V 时用的是 `slot_mapping`。

每次模型生成新 token 的 K/V 向量后，需要把它们写入 KV cache 的正确物理位置。vLLM 用一个 **Triton kernel** 预计算每个 token 对应的 slot：

```python
# vllm/v1/worker/block_table.py — Triton kernel（简化版）
slot_id = block_number * block_size + offset_within_block
```

其中 `block_number` 来自 block_table 查表，`offset_within_block` 是 token 在 block 内的位置。计算好的 `slot_mapping` 传给 `reshape_and_cache_flash`（C++ 算子），它会把新的 K/V 向量 **scatter** 到 KV cache 的正确物理位置。

::: tip 读写分离
总结一下 block_table 在 attention 中的两个用途：
- **写入 KV cache**：`slot_mapping[token_idx]` → 每个 token 的 K/V 写到哪个 slot
- **读取 KV cache**：`block_table[seq_idx][logical_block]` → attention 从哪些物理 block 读 K/V

两者用的底层映射信息相同（都来自 Scheduler 的逻辑→物理映射），但表示形式不同，各自优化了自己的访问模式。
:::

## KV Cache 管理：调度层的实现

上一节讲了 block_table 在 attention 层如何使用。这一节看 Scheduler 层如何管理这些 block 的分配和回收。

### BlockPool：物理 block 的分配器

`BlockPool`（`vllm/v1/core/block_pool.py`）管理所有物理 KV cache block：

- 维护一个 **空闲队列**（`FreeKVCacheBlockQueue`），使用 LRU 淘汰策略
- 每个 block 有 **引用计数**（`ref_cnt`），支持多个请求共享同一个物理 block
- 引用计数降为 0 的 block 不会立即被释放——它留在缓存中等待可能的前缀命中（prefix cache hit），直到内存压力触发 LRU 淘汰

### KVCacheManager：逻辑到物理的映射

`KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）实现了第一篇文章中的 block table 映射：

| 操作 | 方法 | 说明 |
|---|---|---|
| 查询缓存 | `get_computed_blocks()` | 对新请求查找最长前缀缓存命中 |
| 分配 block | `allocate_slots()` | 为新 token 分配物理 block |
| 释放 block | `free()` | 减少引用计数，block 进入 LRU 队列 |
| Copy-on-Write | 在 `allocate_slots()` 内部 | 当共享 block 需要写入时复制 |

### Prefix Caching（自动前缀缓存）

这是 v1 架构默认启用的重要优化（`enable_prefix_caching=True`）。核心思想：

> 如果两个请求有相同的 prompt 前缀，它们的 KV cache 中对应的 block 内容是完全一样的。那为什么要算两遍？

实现方式：
1. 每个 **完全填满** 的 block 的内容会被 hash（默认 SHA-256）
2. `BlockPool` 维护一个 `hash → physical_block` 的映射
3. 新请求到来时，`get_computed_blocks()` 对其 token 序列逐 block 计算 hash，查找已缓存的 block
4. 命中的 block 直接复用（增加引用计数），跳过对应的 KV 计算

这在多用户使用相同 system prompt 的场景下极为有效——system prompt 的 KV cache 只需要计算一次。

## 关键控制参数

理解这些参数对于调优 vLLM 部署至关重要。

### 调度参数（SchedulerConfig）

| 参数 | 默认值 | 作用 |
|---|---|---|
| `max_num_batched_tokens` | 2048 | 每个 iteration 最多处理的 token 总数 |
| `max_num_seqs` | 128 | 每个 iteration 最大并发序列数 |
| `policy` | `"fcfs"` | 调度策略：`"fcfs"`（先来先服务）或 `"priority"` |
| `enable_chunked_prefill` | `True` | 是否允许将长 prefill 拆分到多个 iteration |
| `max_num_partial_prefills` | 1 | 同时进行的部分 prefill 最大数量 |
| `long_prefill_token_threshold` | 0 | 超过此长度的 prefill 被视为"长请求" |

::: details max_num_batched_tokens 的影响
这是影响吞吐和延迟平衡的最关键参数。

- **调高**（如 8192）：每轮处理更多 token → 高吞吐，但单请求延迟可能增加
- **调低**（如 1024）：每轮 token 少 → 低延迟，但吞吐受限
- 与 `enable_chunked_prefill` 配合：启用 chunked prefill 时，长 prompt 不会独占整个 iteration，可以和 decode 请求混合执行
:::

### 缓存参数（CacheConfig）

| 参数 | 默认值 | 作用 |
|---|---|---|
| `block_size` | 自动 | 每个 KV block 包含的 token 数（影响内存碎片和 swap 效率） |
| `gpu_memory_utilization` | 0.9 | GPU 显存利用率上限（留 10% 余量给 CUDA 运行时） |
| `enable_prefix_caching` | `True` | 启用自动前缀缓存 |
| `cache_dtype` | `"auto"` | KV cache 数据类型（可设为 `"fp8"` 减少显存占用） |

::: details gpu_memory_utilization 怎么生效？
vLLM 启动时会执行一次 **内存 profiling**：

1. Worker 调用 `determine_available_memory()`
2. 加载模型后，测量剩余 GPU 显存
3. 用 `gpu_memory_utilization × total_gpu_memory - model_memory` 计算可用于 KV cache 的显存
4. 用可用显存 ÷ 每个 block 的大小 = 物理 block 总数

这个数字直接决定了系统能同时服务多少请求。
:::

### 并行参数（ParallelConfig）

| 参数 | 默认值 | 作用 |
|---|---|---|
| `tensor_parallel_size` | 1 | Tensor Parallelism 并行度（切分到多少张 GPU） |
| `pipeline_parallel_size` | 1 | Pipeline Parallelism 阶段数 |
| `data_parallel_size` | 1 | Data Parallelism 副本数 |
| `distributed_executor_backend` | 自动 | 分布式后端：`"mp"`（多进程）或 `"ray"` |

### 采样参数（SamplingParams）

| 参数 | 默认值 | 作用 |
|---|---|---|
| `temperature` | 1.0 | 控制随机性（0 = greedy，越高越随机） |
| `top_p` | 1.0 | Nucleus sampling 阈值 |
| `top_k` | 0 | Top-k 采样（0 = 不启用） |
| `max_tokens` | 16 | 最大生成 token 数 |
| `n` | 1 | 每个 prompt 生成的候选序列数（parallel sampling） |
| `presence_penalty` | 0.0 | 存在惩罚（抑制重复话题） |
| `frequency_penalty` | 0.0 | 频率惩罚（抑制重复 token） |

## 请求的状态机

一个请求在 vLLM 内部的状态转换：

```
                                ┌──────────────┐
                                │   WAITING    │ ◄─── 新请求到达
                                └──────┬───────┘
                                       │ Scheduler 分配资源
                                       ▼
                                ┌──────────────┐
            ┌──────────────────►│   RUNNING    │
            │                   └──┬───────┬───┘
            │                      │       │
            │       生成 token 继续 │       │ 触发停止条件
            │                      │       ▼
            │                      │  ┌──────────────┐
            │                      │  │  FINISHED_*  │ ──► 释放 KV cache，返回结果
            │                      │  └──────────────┘
            │                      │
            │                      │ 显存不足
            │                      ▼
            │               ┌──────────────┐
            │               │  PREEMPTED   │
            │               └──────┬───────┘
            │                      │ 重新排队到 waiting 头部
            └──────────────────────┘
```

**FINISHED 的几种子状态**：
- `FINISHED_STOPPED`：正常完成（命中 stop 条件）
- `FINISHED_LENGTH_CAPPED`：达到 max_tokens 或 max_model_len
- `FINISHED_ABORTED`：被客户端取消或系统中止

## Chunked Prefill：拆分长 prompt

这是 v1 架构默认启用的优化，解决一个实际问题：

> 如果一个请求的 prompt 有 10000 个 token，在传统实现中，这个 prefill 会独占整个 iteration，其他所有 decode 请求都必须等待。这会导致所有正在生成的请求出现延迟尖刺。

Chunked Prefill 的做法：

1. 将长 prompt 拆分为 `max_num_batched_tokens` 大小的 chunk
2. 每个 iteration 只处理一个 chunk，剩余部分放到下一个 iteration
3. 同一个 iteration 中，可以同时处理这个 prefill chunk 和其他请求的 decode token

```
传统方式（没有 chunked prefill）:
  iter 1: [======= 10000 token prefill =======]  ← 其他请求全部等待
  iter 2: [decode A] [decode B] [decode C]

Chunked Prefill（max_num_batched_tokens=2048）:
  iter 1: [2048 prefill chunk] [decode A] [decode B]  ← 并行执行
  iter 2: [2048 prefill chunk] [decode A] [decode B]
  iter 3: [2048 prefill chunk] [decode A] [decode B]
  ...
```

这样，正在生成的请求不会因为新请求的长 prompt 而被饿死。

<HtmlVisualization
  src="/machine-learning/inference/visualizations/chunked-prefill.html"
  height="580px"
  title="传统 Prefill vs Chunked Prefill"
/>

## Speculative Decoding：投机解码

vLLM 支持多种投机解码策略（Ngram、Eagle、Medusa）。核心思想：

1. **Draft 阶段**：用一个快速的小模型（或 n-gram 匹配）猜测接下来的 k 个 token
2. **Verify 阶段**：用目标大模型并行验证这 k 个 token
3. 被验证通过的 token 直接采纳，失败的 token 从失败点回退

在代码中：
- `Request.spec_token_ids` 存储 draft token
- `SpecDecodeMetadata` 协调验证过程
- `RejectionSampler` 执行验证采样
- 验证失败时，`num_computed_tokens` 回退到验证失败的位置

## 离线使用 vs 在线 Serving

vLLM 同时支持两种使用模式，共享同一套 Scheduler + Worker 基础设施：

### 离线批量推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-8B")
outputs = llm.generate(["Hello", "World"], SamplingParams(max_tokens=100))
```

调用链：`LLM.generate()` → `LLMEngine` → `EngineCoreClient` → 同步循环直到所有请求完成。

### 在线 Serving

```bash
vllm serve Qwen/Qwen3-8B --port 8000
```

调用链：`FastAPI` → `AsyncLLM` → `EngineCoreClient` → 异步流式返回。

两者的区别仅在于 engine 层的封装方式——离线用同步的 `LLMEngine`，在线用异步的 `AsyncLLM`，底层的 EngineCore、Scheduler、Worker 完全相同。

## 总结：各组件的职责

| 组件 | 文件位置 | 核心职责 |
|---|---|---|
| **AsyncLLM** | `vllm/v1/engine/async_llm.py` | API 进程的引擎客户端，输入/输出处理 |
| **EngineCore** | `vllm/v1/engine/core.py` | 主循环：schedule → execute → update |
| **Scheduler** | `vllm/v1/core/sched/scheduler.py` | 每轮决策：哪些请求运行，分配多少 token |
| **KVCacheManager** | `vllm/v1/core/kv_cache_manager.py` | Block table 映射，前缀缓存，Copy-on-Write |
| **BlockPool** | `vllm/v1/core/block_pool.py` | 物理 block 分配，引用计数，LRU 淘汰 |
| **Executor** | `vllm/v1/executor/` | 将计算任务分发给 Worker |
| **Worker** | `vllm/v1/worker/gpu_worker.py` | 每 GPU 一个，执行模型推理 |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | 输入准备、模型前向、采样、CUDA Graph |

从一个请求的视角来看：**进入 → 排队 → 被 Scheduler 选中 → 分配 KV block → Worker 执行前向传播 → 采样 token → 检查是否完成 → 返回结果或继续下一轮**。这个循环每秒执行数十到数百次，驱动着 vLLM 的高吞吐推理引擎。
