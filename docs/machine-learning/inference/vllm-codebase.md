---
date: 2026-03-28
title: "vLLM Codebase Deep Dive: Tracing a Request from API to Token Output"
description: Deep analysis of vLLM's code architecture, tracing an inference request through the complete call chain from API entry to token output — Scheduler, Worker, KV Cache Manager, and more
tags: [vllm, inference, source-code, scheduler, kv-cache, serving]
---

# vLLM Codebase Deep Dive: Tracing a Request from API to Token Output

> Based on the vLLM v1 architecture (main branch as of March 2026). Source: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm). Recommended prerequisite: [Part 1: vLLM & PagedAttention](./vllm-pagedattention) for the algorithmic foundations.

In Part 1, we understood **what** PagedAttention does from the paper's perspective. In this article, we go into the source code to see **how** vLLM actually implements it — what components a request passes through, and what happens at each step from API entry to token output.

## Codebase Overview

Top-level directory structure of the vLLM repository:

```
vllm/
├── vllm/                  # Core Python library
│   ├── v1/                # ⭐ v1 architecture (current focus)
│   │   ├── engine/        #    Engine layer: AsyncLLM, EngineCore, LLMEngine
│   │   ├── core/          #    Scheduling layer: Scheduler, KVCacheManager, BlockPool
│   │   ├── worker/        #    Execution layer: Worker, GPUModelRunner
│   │   └── executor/      #    Distribution layer: UniprocExecutor, MultiprocExecutor
│   ├── engine/            # Legacy engine (being phased out)
│   ├── entrypoints/       # API endpoints: OpenAI-compatible server, CLI, offline batch
│   ├── config/            # Configuration modules (27 config classes)
│   ├── model_executor/    # Model loading, ops, model implementations
│   ├── distributed/       # Distributed communication primitives
│   ├── multimodal/        # Multi-modal input (images, audio, video)
│   └── sampling_params.py # Sampling parameters
├── csrc/                  # C++/CUDA custom kernels
├── benchmarks/            # Performance benchmarks
├── tests/                 # Test suite
└── examples/              # Usage examples
```

::: info v1 Architecture vs Legacy
The vLLM codebase contains two coexisting architectures: the legacy engine under `vllm/engine/` (monolithic design) and the new v1 architecture under `vllm/v1/` (multi-process design). v1 is the active development focus, and this article covers only v1.
:::

## Architecture Overview: Three Process Types

The v1 architecture's most distinctive feature is **multi-process separation**. A vLLM serving instance consists of three types of processes:

```
┌─────────────────────────────────────────────────────────┐
│                    API Server Process                    │
│                                                         │
│  FastAPI (OpenAI-compatible API)                        │
│       │                                                 │
│       ▼                                                 │
│  AsyncLLM                                               │
│   ├─ InputProcessor   (request → EngineCoreRequest)     │
│   ├─ OutputProcessor  (EngineCoreOutputs → streaming)   │
│   └─ Detokenizer      (incremental detokenization)      │
│       │                                                 │
│       │ ZMQ IPC                                         │
├───────┼─────────────────────────────────────────────────┤
│       ▼             EngineCore Process                   │
│  EngineCoreProc                                         │
│   ├─ Scheduler       (decides what to run each step)    │
│   ├─ KVCacheManager  (manages KV block alloc/free)      │
│   └─ Executor        (dispatches work to Workers)       │
│       │                                                 │
│       │ SharedMemory MessageQueue                       │
├───────┼─────────────────────────────────────────────────┤
│       ▼             Worker Processes (one per GPU)       │
│  Worker                                                 │
│   └─ GPUModelRunner  (prepare inputs + model forward)   │
└─────────────────────────────────────────────────────────┘
```

**Why multi-process?** Python's GIL (Global Interpreter Lock) serializes CPU-intensive threads. Separating API handling, scheduling, and GPU computation into different processes avoids GIL contention and achieves true parallelism.

Inter-process communication uses two mechanisms:
- **ZMQ IPC**: API Server process ↔ EngineCore process
- **SharedMemory MessageQueue**: EngineCore process ↔ Worker processes (zero-copy, ultra-low latency)

## Complete Request Lifecycle

Let's trace a request from entry to response.

### Stop 1: API Entry

A user sends an OpenAI-format request:

```bash
curl http://localhost:8000/v1/completions \
  -d '{"model": "Qwen/Qwen3-8B", "prompt": "Hello", "max_tokens": 100}'
```

The request hits the FastAPI route layer (`vllm/entrypoints/openai/api_server.py`), then goes to `AsyncLLM`:

```
HTTP Request
  → FastAPI route handler
    → AsyncLLM.generate()
```

`AsyncLLM` (`vllm/v1/engine/async_llm.py`) is the core class in the API Server process. It does two things:

1. **InputProcessor** converts the raw request into an `EngineCoreRequest` (token IDs, sampling params, etc.)
2. Sends the `EngineCoreRequest` to the EngineCore process via ZMQ

### Stop 2: EngineCore Main Loop

EngineCore (`vllm/v1/engine/core.py`) runs in a separate process, executing a never-ending busy loop:

```python
# vllm/v1/engine/core.py — simplified main loop
def run_busy_loop(self):
    while True:
        # 1. Read new requests
        self._process_input_queue()

        # 2. Execute one inference step
        outputs = self._process_engine_step()

        # 3. Send results back to API Server process
        self._process_output_queue(outputs)
```

Each loop iteration is one **iteration**. `_process_engine_step()` is the core:

```python
def _process_engine_step(self):
    # Step 1: Scheduler decides what to run
    scheduler_output = self.scheduler.schedule()

    # Step 2: Executor dispatches to Workers
    model_output = self.executor.execute_model(scheduler_output)

    # Step 3: Scheduler processes output, updates state
    engine_core_outputs = self.scheduler.update_from_output(model_output)

    return engine_core_outputs
```

These three steps are the heartbeat of vLLM's inference engine.

### Stop 3: Scheduler — The Decision Maker

The Scheduler (`vllm/v1/core/sched/scheduler.py`) is vLLM's brain. It maintains two queues:

- **`running`**: currently executing requests
- **`waiting`**: queued requests awaiting scheduling

Each `schedule()` call has **two phases**:

#### Phase 1: Schedule Running Requests

For each request in the `running` list:

1. Calculate tokens needed this iteration: `num_tokens = num_tokens_with_spec - num_computed_tokens`
2. Allocate KV cache blocks via `KVCacheManager.allocate_slots()`
3. If allocation fails (out of memory) → **preempt** the lowest-priority running request, free its blocks, retry

#### Phase 2: Schedule Waiting Requests

Only runs if Phase 1 had no preemptions:

1. Pick requests from `waiting` queue by policy (FCFS or Priority)
2. Query prefix cache hits: `KVCacheManager.get_computed_blocks()`
3. Calculate new tokens to schedule, check token budget
4. Allocate KV cache, move request to `running`

Key constraints:
- **`max_num_batched_tokens`**: per-iteration token cap
- **`max_num_seqs`**: max concurrent sequences per iteration
- **`max_model_len`**: maximum supported sequence length

<HtmlVisualization
  src="/machine-learning/inference/visualizations/scheduler-two-phase.html"
  height="660px"
  title="Scheduler Two-Phase Scheduling"
/>

::: tip Continuous Batching
Unlike static batching (where a batch must wait for all requests to finish), vLLM's Scheduler **re-decides every iteration**. Requests can join or leave the batch at any iteration. This is Continuous Batching — one of the keys to vLLM's high throughput.
:::

<HtmlVisualization
  src="/machine-learning/inference/visualizations/continuous-batching.html"
  height="620px"
  title="Static Batching vs Continuous Batching"
/>

### Stop 4: Executor — Dispatching Computation

The Executor dispatches `SchedulerOutput` to GPU Workers. Different implementations for different deployments:

| Executor | Use Case |
|---|---|
| `UniprocExecutor` | Single process, single GPU |
| `MultiprocExecutor` | Single node, multi-GPU (multi-process) |
| `RayExecutor` | Multi-node, multi-GPU (via Ray) |

### Stop 5: Worker + GPUModelRunner — Actual Inference

The Worker (`vllm/v1/worker/gpu_worker.py`) delegates to `GPUModelRunner` (`vllm/v1/worker/gpu_model_runner.py`):

1. **Prepare input tensors**: Build input_ids, position_ids, attention metadata from `SchedulerOutput`
2. **Model forward pass**: Call the model's `forward()` method
3. **Sampling**: Sample next token from logits based on `SamplingParams`
4. **Async output transfer**: Use a separate CUDA stream to transfer results from GPU to CPU

`GPUModelRunner` also manages **CUDA Graphs** — pre-captured computation graphs for different batch sizes that are replayed during inference, avoiding kernel launch overhead.

### Stop 6: Result Return

```
ModelRunnerOutput (Worker)
  → SharedMemory MQ → Executor
    → Scheduler.update_from_output()
      → Check stop conditions (EOS, stop strings, max_tokens)
      → Update request state
      → Free completed requests' KV cache
    → EngineCoreOutputs
      → ZMQ → AsyncLLM
        → OutputProcessor.process_outputs()
          → Detokenizer (incremental detokenization)
          → RequestOutput
            → Stream back to client
```

## PagedAttention in Code

Part 1 explained the PagedAttention algorithm. How does "reading K/V by block" actually work in the code?

### Two Implementations: CUDA Kernel and FlashAttention

vLLM actually has **two** PagedAttention implementations:

| Implementation | Location | Used By |
|---|---|---|
| **Custom CUDA Kernel** | `csrc/attention/attention_kernels.cuh` | Legacy engine (v0) |
| **FlashAttention / FlashInfer** | `vllm/v1/attention/backends/` | v1 engine (current) |

In the v1 architecture, vLLM passes the block_table directly to FlashAttention/FlashInfer instead of implementing the attention kernel itself. But understanding the custom kernel is more instructive for grasping how PagedAttention works, so we'll look at both.

### Custom CUDA Kernel: Manual Block Table Lookup

Core code in `csrc/attention/attention_kernels.cuh`, simplified kernel signature:

```cpp
__device__ void paged_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq] ← PAGE TABLE!
    const int* __restrict__ seq_lens,     // [num_seqs]
    ...)
```

**The critical operation — page table lookup:**

```cpp
// Get this sequence's page table row
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// Iterate over logical blocks
for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
    // Lookup: logical block → physical block
    const int64_t physical_block_number = block_table[block_idx];

    // Compute K's memory address using the physical block number
    const cache_t* k_ptr = k_cache
        + physical_block_number * kv_block_stride
        + kv_head_idx * kv_head_stride;

    // Compute qk = scale * dot(q, k)
    // ...
}
```

This is the actual code behind the "logical block → physical block" mapping from Part 1. `block_table[block_idx]` is the "page table lookup" — given a logical block index, return the physical block number, then use it to read actual K/V data from GPU memory.

**K and V have different memory layouts**, optimized for their respective access patterns:

```
K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    ↑ Interleaved layout for coalesced memory access during dot(q, k)

V cache: [num_blocks, num_kv_heads, head_size, block_size]
    ↑ Transposed layout for efficient weighted sum: output += score * v
```

::: details V1 Kernel vs V2 Kernel
The custom kernel has two versions:
- **V1**: Each thread block handles the **entire** sequence for one head. Good for short sequences.
- **V2**: **Partitions** the sequence across multiple thread blocks, then merges with a reduce kernel (using the online softmax trick to combine independently computed softmax partitions). Good for long sequences.
:::

### v1 Architecture: FlashAttention's Paged KV Support

In the v1 architecture, vLLM uses FlashAttention's (or FlashInfer's) native paged KV cache support. Core code in `vllm/v1/attention/backends/flash_attn.py`:

```python
# vllm/v1/attention/backends/flash_attn.py — simplified
flash_attn_varlen_func(
    q=query,
    k=key_cache,              # The entire KV cache pool
    v=value_cache,
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    block_table=block_table,   # ← Page table passed directly to FlashAttention!
    softmax_scale=self.scale,
    causal=True,
)
```

The same `block_table` tensor is passed directly to FlashAttention. Internally, FlashAttention uses the same logic as the custom kernel — looking up physical block numbers and reading K/V from non-contiguous addresses. The implementation is just encapsulated inside the FlashAttention library.

### Block Table Data Flow

The complete path from Scheduler to Kernel:

```
Scheduler allocates logical → physical block mapping
    │
    ▼ (SchedulerOutput carries block_ids)
GPUModelRunner._update_states()
    │  Write to BlockTable object's CPU buffer
    ▼
BlockTable.commit_block_table()
    │  CPU → GPU copy (DMA transfer)
    ▼
block_table_tensor (int32 tensor on GPU)
    │  shape: [num_reqs, max_blocks_per_req]
    ▼
CommonAttentionMetadata
    │  Wrapped as FlashAttentionMetadata.block_table
    ▼
flash_attn_varlen_func(block_table=...)
    │  FlashAttention reads K/V by block lookup internally
    ▼
Attention output
```

### Slot Mapping: Writing to KV Cache

Reading K/V uses `block_table`, but **writing** new K/V uses `slot_mapping`.

When the model generates new K/V vectors for a token, they need to be written to the correct physical location in the KV cache. vLLM uses a **Triton kernel** to precompute each token's slot:

```python
# vllm/v1/worker/block_table.py — Triton kernel (simplified)
slot_id = block_number * block_size + offset_within_block
```

The computed `slot_mapping` is passed to `reshape_and_cache_flash` (a C++ op) which **scatters** new K/V vectors into the correct physical slots.

::: tip Read-Write Separation
The block_table serves two purposes in attention:
- **Writing KV cache**: `slot_mapping[token_idx]` → which slot to write each token's K/V into
- **Reading KV cache**: `block_table[seq_idx][logical_block]` → which physical blocks attention reads K/V from

Both use the same underlying mapping (from Scheduler), but in different representations optimized for their respective access patterns.
:::

## KV Cache Management: The Scheduling Layer

The previous section covered how block_table is used in the attention layer. This section covers how the Scheduler manages block allocation and recycling.

### BlockPool: Physical Block Allocator

`BlockPool` (`vllm/v1/core/block_pool.py`) manages all physical KV cache blocks:

- Maintains a **free queue** (`FreeKVCacheBlockQueue`) with LRU eviction
- Each block has a **reference count** (`ref_cnt`) for sharing across requests
- Blocks with `ref_cnt == 0` stay cached for potential prefix hits until evicted under memory pressure

### KVCacheManager: Logical-to-Physical Mapping

`KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`) implements the block table from Part 1:

| Operation | Method | Description |
|---|---|---|
| Query cache | `get_computed_blocks()` | Find longest prefix cache hit for new requests |
| Allocate blocks | `allocate_slots()` | Allocate physical blocks for new tokens |
| Free blocks | `free()` | Decrement ref count, block enters LRU queue |
| Copy-on-Write | Inside `allocate_slots()` | Copy shared blocks when write is needed |

### Prefix Caching (Automatic Prefix Caching / APC)

Enabled by default (`enable_prefix_caching=True`). Core idea:

> If two requests share the same prompt prefix, their KV cache blocks for that prefix are identical. Why compute them twice?

Implementation:
1. Each **fully filled** block's content is hashed (SHA-256 by default)
2. `BlockPool` maintains a `hash → physical_block` mapping
3. For new requests, `get_computed_blocks()` computes per-block hashes and looks up cached blocks
4. Hits are reused (increment ref count), skipping KV computation

Extremely effective when many users share the same system prompt.

## Key Control Parameters

### Scheduling (SchedulerConfig)

| Parameter | Default | Effect |
|---|---|---|
| `max_num_batched_tokens` | 2048 | Max tokens per iteration |
| `max_num_seqs` | 128 | Max concurrent sequences per iteration |
| `policy` | `"fcfs"` | Scheduling policy: `"fcfs"` or `"priority"` |
| `enable_chunked_prefill` | `True` | Split long prefills across iterations |
| `max_num_partial_prefills` | 1 | Max concurrent partial prefills |

### Cache (CacheConfig)

| Parameter | Default | Effect |
|---|---|---|
| `block_size` | auto | Tokens per KV block |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction for the executor |
| `enable_prefix_caching` | `True` | Enable automatic prefix caching |
| `cache_dtype` | `"auto"` | KV cache dtype (can use `"fp8"`) |

### Parallelism (ParallelConfig)

| Parameter | Default | Effect |
|---|---|---|
| `tensor_parallel_size` | 1 | TP degree (split across GPUs) |
| `pipeline_parallel_size` | 1 | PP stages |
| `data_parallel_size` | 1 | DP replicas |

### Sampling (SamplingParams)

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | 1.0 | Randomness (0 = greedy) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | 0 | Top-k sampling (0 = disabled) |
| `max_tokens` | 16 | Max generated tokens |
| `n` | 1 | Candidates per prompt (parallel sampling) |

## Chunked Prefill

Default-enabled optimization solving a real problem:

> A 10,000-token prompt would monopolize an entire iteration in traditional systems, causing latency spikes for all other decode requests.

Chunked Prefill splits long prompts into chunks of `max_num_batched_tokens`, processing one chunk per iteration alongside other requests' decode tokens.

```
Traditional (no chunked prefill):
  iter 1: [======= 10000 token prefill =======]  ← all others wait
  iter 2: [decode A] [decode B] [decode C]

Chunked Prefill (max_num_batched_tokens=2048):
  iter 1: [2048 prefill chunk] [decode A] [decode B]  ← concurrent
  iter 2: [2048 prefill chunk] [decode A] [decode B]
  ...
```

<HtmlVisualization
  src="/machine-learning/inference/visualizations/chunked-prefill.html"
  height="580px"
  title="Traditional Prefill vs Chunked Prefill"
/>

## Speculative Decoding

vLLM supports multiple speculative decoding strategies (Ngram, Eagle, Medusa):

1. **Draft phase**: A fast small model (or n-gram matching) guesses the next k tokens
2. **Verify phase**: The target model verifies all k tokens in parallel
3. Verified tokens are accepted; rejected tokens trigger rollback

In code: `Request.spec_token_ids` stores drafts, `RejectionSampler` handles verification, and `num_computed_tokens` rolls back on rejection.

## Summary: Component Responsibilities

| Component | File | Core Responsibility |
|---|---|---|
| **AsyncLLM** | `vllm/v1/engine/async_llm.py` | API process engine client, I/O processing |
| **EngineCore** | `vllm/v1/engine/core.py` | Main loop: schedule → execute → update |
| **Scheduler** | `vllm/v1/core/sched/scheduler.py` | Per-step decisions: which requests, how many tokens |
| **KVCacheManager** | `vllm/v1/core/kv_cache_manager.py` | Block table mapping, prefix cache, CoW |
| **BlockPool** | `vllm/v1/core/block_pool.py` | Physical block allocation, ref counting, LRU |
| **Executor** | `vllm/v1/executor/` | Dispatch computation to Workers |
| **Worker** | `vllm/v1/worker/gpu_worker.py` | Per-GPU inference execution |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | Input prep, model forward, sampling, CUDA Graphs |

From a request's perspective: **enter → queue → selected by Scheduler → allocate KV blocks → Worker executes forward pass → sample token → check completion → return result or continue next iteration**. This loop runs tens to hundreds of times per second, powering vLLM's high-throughput inference engine.
