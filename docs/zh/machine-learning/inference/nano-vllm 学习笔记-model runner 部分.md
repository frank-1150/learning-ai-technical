---
date: "2026-06-14"
title: "nano-vllm 学习笔记（三）：Model Runner 与推理执行"
description: 从第一性原理讲清楚 Model Runner 存在的理由——它如何桥接调度决策与 GPU 计算、为什么按这个顺序初始化、prefill 和 decode 的数据格式为什么必须不同、CUDA graph 解决的是哪个问题、以及多 GPU 下的 SharedMemory 协议如何工作。
tags: [inference, vllm, model-runner, cuda-graph, kv-cache, tensor-parallel, nano-vllm, flash-attention]
---

# nano-vllm 学习笔记（三）：Model Runner 与推理执行

> 前两篇讲了 Block Manager（管物理显存）和 Scheduler（做调度决策）。这篇对应 `nanovllm/engine/model_runner.py`，讲调度结果如何被转化成一次真实的 GPU forward。

## 为什么需要 Model Runner

先问最基本的问题：Scheduler 已经决定了"这一步处理哪些 sequence"，为什么还需要一个单独的 Model Runner？Scheduler 直接调用 `model.forward()` 不行吗？

不行，因为 Scheduler 的决策是"逻辑层面的"——它知道序号、token 数量、block_table——但它不知道如何把这些信息转化成 GPU 能消费的 tensor 格式。而这件事并不简单：

- **Prefill 阶段**：多个 sequence 长度各不相同，需要拼成一个"不等长 batch"，Flash Attention 要求 `cu_seqlens`（每个 sequence 在拼接后的起止位置）；新计算出的 K/V 需要写进 KV cache 的正确位置（由 `slot_mapping` 指定）；如果有 prefix cache 命中，attention 要跨 block 查历史 K/V。
- **Decode 阶段**：每个 sequence 只有 1 个新 token，但它要 attend 到历史上数百甚至数千个 token，这些 token 的 K/V 散落在不同的 physical block 里，Flash Attention 需要 `block_tables` 来寻址。
- **CUDA Graph**：Decode 阶段为了消除 Python dispatch overhead，用 CUDA Graph 把整个 forward 录成一段 CUDA 指令流，每次执行只需 "replay"——但这要求 tensor shape 固定，且需要提前为所有可能的 batch size 都 capture 一份 graph。

这些工作——格式转换、tensor 准备、CUDA Graph 管理、多 GPU 通信——都是调度层不该管的事。Model Runner 是这道"逻辑到物理"的翻译层。

一句话描述它的职责：**把 Scheduler 输出的 Sequence 列表，翻译成 GPU forward 所需的 tensor，执行推理，返回新 token 的 ID。**

---

## 初始化：为什么按这个顺序

`ModelRunner.__init__` 按固定顺序做了四件事：

```python
self.model = Qwen3ForCausalLM(hf_config)   # 1. 加载模型
load_model(self.model, config.model)
self.sampler = Sampler()
self.warmup_model()                          # 2. warmup
self.allocate_kv_cache()                     # 3. 分配 KV cache
if not self.enforce_eager:
    self.capture_cudagraph()                 # 4. capture CUDA graph
```

顺序不是随意的，有严格的依赖关系。

### 第一步：加载模型

```python
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

`set_default_device("cuda")` 让后续所有 `torch.empty` / `torch.zeros` 默认分配在 GPU 上——这样加载模型权重时无需显式指定设备。模型加载完之后 GPU 显存里已经占了一大块（几 GB 到几十 GB，取决于模型大小）。

### 第二步：warmup，量出 peak 显存

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)    # 一次最大规模的 prefill
    torch.cuda.empty_cache()
```

Warmup 的目的是：**跑一次最大规模的 prefill，让 PyTorch 把各种 allocator 的状态稳定下来，同时记录峰值显存使用量**。

为什么要这么做？因为 `allocate_kv_cache()` 的核心任务是"用剩余显存分配 KV cache block"——但"剩余"是相对于"推理实际需要多少显存"来说的。如果不先跑一次 warmup，我们不知道 PyTorch 在最坏情况下（最大 batch、最长序列）会临时分配多少 activation 显存。直接用 `total - model_weight` 估算会严重高估剩余空间，导致 KV cache 占完显存，推理时 OOM。

Warmup 之后 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` 就是那个"最坏情况激活显存"的精确测量值。

### 第三步：根据实测结果分配 KV cache

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    config.num_kvcache_blocks = int(
        total * gpu_memory_utilization - used - peak + current
    ) // block_bytes
    self.kv_cache = torch.empty(2, num_hidden_layers, num_kvcache_blocks,
                                block_size, num_kv_heads, head_dim)
```

这里的公式值得细读：

```
可用显存 = total × gpu_memory_utilization - used - (peak - current)
```

- `total × gpu_memory_utilization`：总可用上限（比如 90% 的 80GB = 72GB）
- `- used`：当前已用（模型权重 + PyTorch 底层 allocator 开销）
- `- (peak - current)`：预留给 forward 期间临时激活的额外空间，用 warmup 时的 peak-current 来估算

**KV cache 的形状**：`[2, L, N, B, H, D]`——2 表示 K 和 V，L 是层数，N 是 block 总数，B 是每个 block 的 token 数（block_size），H 是 KV head 数，D 是 head dim。之所以一次性分配好整个大 tensor 而不是按需分配，是为了让所有 block 在物理上连续，Flash Attention 可以直接通过 block_id 和 offset 寻址，不需要指针跳转。

分配完之后，把每个 attention layer 的 `k_cache` / `v_cache` 属性指向这个大 tensor 的对应切片：

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

这样每个 attention layer 的 K/V 写入都直接落进统一管理的显存池。

### 第四步：capture CUDA Graph（仅 decode 阶段）

CUDA Graph 是这节最有意思的部分，后面专门展开讲。

---

## Context：一个推理步的全局上下文

在读 `prepare_prefill` 和 `prepare_decode` 之前，先理解一个关键设计：`Context` 对象。

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
```

它是一个**进程级单例**：`set_context(...)` 设置，`get_context()` 读取，`reset_context()` 清空。

为什么要这样设计，而不是直接把这些 tensor 作为参数传进 `model.forward()`？

原因在于模型的调用链太深：`ModelRunner.run_model()` → `model()` → 多个 `DecoderLayer` → 多个 `Attention.forward()`。如果要把 `cu_seqlens`、`slot_mapping`、`block_tables` 一路作为参数传下去，每个中间层都要接受并透传这些参数，代码会很啰嗦，而且这些参数和层本身的逻辑无关。

用全局 Context 相当于一个"推理步配置"——每次 `run()` 开始时设置好，所有 attention layer 直接从全局读，`run()` 结束时清空。这是一种**进程内的 request-scoped 配置模式**。

---

## 两种模式的数据准备

### prepare_prefill：变长 batch，flash_attn_varlen_func

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    slot_mapping = []
    block_tables = None
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        ...
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:   # prefix cache 命中
        block_tables = self.prepare_block_tables(seqs)
```

几个关键点：

**`cu_seqlens_q` 和 `cu_seqlens_k` 为什么要分开？**

Flash Attention 的 `flash_attn_varlen_func` 支持 Q 和 KV 序列长度不同。在有 prefix cache 的情况下：
- `seqlen_q`（query 长度）= 这轮要计算的 token 数（`num_scheduled_tokens`）
- `seqlen_k`（key/value 长度）= 从位置 0 到当前 token 的完整历史（`end`）

因为 prefix cache 中已有的 K/V 不需要重新算，但 attention 计算时 Q 要 attend 到它们。所以 Q 更短，KV 更长。`cu_seqlens_k[-1] > cu_seqlens_q[-1]` 就是检测"是否有任何 sequence 命中了 prefix cache"。

如果没有 prefix cache 命中，`seqlen_q == seqlen_k`，两者相等，就不需要 block_tables。

**slot_mapping：新 K/V 写去哪里**

Prefill 完新 token 的 K/V 必须写进 KV cache，供将来 decode 阶段使用。`slot_mapping` 是一个 `[num_new_tokens]` 的 int32 tensor，每个元素是这个 token 的 K/V 应该存到哪个 physical slot：

```python
slot = block_id * block_size + offset_within_block
```

Triton kernel 用这个 slot 直接寻址 KV cache：

```python
slot = tl.load(slot_mapping_ptr + idx)
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
```

**`pin_memory=True` + `.cuda(non_blocking=True)`**

所有 tensor 先在 CPU 上分配 pinned memory（固定不换页的物理内存），然后异步传到 GPU。Pinned memory 让 CPU→GPU 的 DMA 传输不经过内核虚拟内存管理，速度更快，且 `non_blocking=True` 让 CPU 不等待传输完成就继续准备下一个 tensor，传输和 CPU 工作可以重叠。

### prepare_decode：定长 batch，flash_attn_with_kvcache

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )
    block_tables = self.prepare_block_tables(seqs)
```

Decode 阶段的特征：每个 sequence 只有 **1 个新 token**。

- `input_ids`：每个 sequence 的最新 token
- `positions`：该 token 在序列里的位置（即 `len(seq) - 1`）
- `context_lens`：这个 sequence 已有多少 token（包括刚生成的这个），Flash Attention 用它知道对每个 sequence 要 attend 多远的历史
- `slot_mapping`：新 token 的 K/V 写到最后一个 block 的当前末尾位置
- `block_tables`：每个 sequence 的 block_table，形状 `[num_seqs, max_num_blocks]`，Flash Attention 用它在做 attention 时按 block_id 寻址历史 K/V

这里与 prefill 的一个重大区别：`block_tables` 在 decode 阶段**永远需要**，而 prefill 只有在有 prefix cache 命中时才需要。原因是 decode 的历史 K/V 本来就散落在多个 block 里，必须通过 block_table 来找；而 prefill 如果没有 prefix cache，K/V 是刚刚算出来的，顺序存入 slot，可以直接用 `slot_mapping` 写入，不需要回头去查 block_table。

---

## run()：一次推理的完整数据流

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

数据流可以这样理解：

```
Sequence 列表
    ↓ prepare_prefill / prepare_decode
input_ids, positions（+ Context 全局状态）
    ↓ run_model
logits [num_seqs, vocab_size]
    ↓ sampler（只在 rank 0）
token_ids [num_seqs]
```

**为什么 `temperatures` 和 `token_ids` 只在 rank 0 处理？**

在 Tensor Parallel 下，每个 rank 都做同样的 forward（模型分片不同），但 logits 最终在每个 rank 上都是完整的（最后 all-gather 到一起）。Sampling 只需要做一次——交给 rank 0，其他 rank 的 `token_ids` 直接返回 None。Scheduler 在主进程（rank 0）里，只消费 rank 0 的返回值。

**Sampler 的实现**

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

这里用的是 **Gumbel-max trick**：对概率分布除以 Gumbel 噪声（指数分布的倒数），然后取 argmax。它在数学上等价于按概率采样，但避免了显式的 `torch.multinomial`（后者在 GPU 上效率较低）。`@torch.compile` 让这段逻辑被 TorchInductor 编译成融合 kernel，减少 kernel launch 次数。

---

## CUDA Graph：消除 decode 的 Python 开销

### 为什么 decode 需要 CUDA Graph

LLM 推理有一个不对称性：prefill 一次处理几百到几千个 token，GPU 计算量大，Python overhead 可以忽略不计；decode 每步只处理 1 个 token per sequence，GPU 计算量极小，反而是 Python/PyTorch dispatch 的固定开销占了相当比例的 step 时间。

每次 `model.forward()` 都需要 Python 解释器逐一遍历所有模块、调用每个 operator 的 dispatch 逻辑、向 CUDA driver 提交 kernel launch 请求。对于几十层的 Transformer，这会是几百个 kernel launch，对应几百次 Python 函数调用和 C++ dispatch。在 decode 阶段，GPU kernel 本身可能只运行几十微秒，但这些 dispatch 开销累加起来可能接近甚至超过 GPU 计算时间。

CUDA Graph 的思路是：**把这一系列 kernel launch 预先录制成一个 graph，replay 时直接向 CUDA driver 提交整个 graph，绕过所有 Python dispatch**。

### capture_cudagraph 的实现

```python
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], ...)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
```

**为什么要为多个 batch size 各 capture 一份 graph？**

CUDA Graph 要求 tensor shape 固定。Decode 阶段每步的 batch size 会变化（随着请求陆续完成），不同 batch size 的 `input_ids`、`slot_mapping` 等 tensor shape 不同，必须分别 capture。`graph_bs` 是一个 bucket 列表：`[1, 2, 4, 8, 16, 32, ...]`，实际 batch size 向上取最近的 bucket。

**`graph_pool`**：多个 graph 共享同一个 CUDA memory pool，避免每个 graph 各自维护独立的内存，降低总显存开销。`reversed(self.graph_bs)` 从大到小 capture，最大 graph 先建，它的 pool 最大；后续更小的 graph 复用同一个 pool，不需要额外分配。

**replay 时如何更新输入**

CUDA Graph capture 的是 kernel sequence 和那一刻的 tensor 指针。Replay 时不能换指针——但可以**更新指针指向的内容**。这就是 `graph_vars` 的作用：

```python
self.graph_vars = dict(
    input_ids=input_ids,
    positions=positions,
    slot_mapping=slot_mapping,
    ...
    outputs=outputs,
)
```

Replay 前，把当前 batch 的数据写进 `graph_vars` 里的 tensor（原地更新），然后 replay：

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
...
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**哪些情况不用 CUDA Graph？**

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # use CUDA graph
```

- Prefill：变长 batch，shape 不固定，无法 replay
- `enforce_eager=True`：用户主动禁用（调试场景）
- `bs > 512`：超出 graph 覆盖的 bucket 范围，fallback 到 eager

---

## Tensor Parallel 下的 SharedMemory 协议

当 `tensor_parallel_size > 1` 时，多个进程需要同步执行。nano-vllm 选择了一种极其简洁的 IPC 方案：**POSIX Shared Memory + Event**。

### 进程角色

```
Rank 0（主进程）：LLMEngine → Scheduler → ModelRunner(rank=0)
Rank 1+（worker 进程）：ModelRunner(rank=i) → loop()
```

Rank 0 是决策者：它运行 Scheduler，决定每步处理哪些 sequence，然后通过 SharedMemory 把方法名和参数广播给其他 rank。

Rank 1+ 是纯执行者：它们跑一个无限循环，等待 rank 0 写入 SharedMemory，然后执行对应的方法：

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

### 通信协议

SharedMemory 是一块 1MB 的共享物理内存，格式非常简单：

```
[0:4]    4 bytes，消息体长度 n（little-endian int）
[4:n+4]  pickle 序列化的 [method_name, *args]
```

**写（rank 0 → others）**：
```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()    # 通知所有 worker
```

**读（rank 1+ 等待）**：
```python
def read_shm(self):
    self.event.wait()    # 阻塞，直到 rank 0 set event
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

`Event` 是 multiprocessing 同步原语（基于 semaphore）：`event.set()` 唤醒等待的进程，`event.wait()` 阻塞直到被 set，`event.clear()` 重置。

**为什么不用 `torch.distributed` 的 broadcast？**

TP 框架里 `dist.broadcast` 是 GPU-to-GPU 通信，用于同步 tensor。但这里传的是"调度指令"（method name、Sequence 对象），是 CPU 端的 Python 对象，不是 GPU tensor。用 SharedMemory + pickle 是更自然的选择——轻量、零拷贝（在同一台机器上）、不依赖 NCCL。

**为什么 `run()` 里 `seqs` 可以直接传**：

这里传的 `seqs` 是经过 `Sequence.__getstate__` 压缩过的轻量 tuple（详见笔记二），pickle 后体积很小，走 SharedMemory 代价可控。

### NCCL 的角色

SharedMemory 只负责"告诉所有 rank 该做什么"。实际的模型 forward 中，每个 rank 只持有部分权重（Tensor Parallel 切分），forward 过程中需要通过 **NCCL** 做 all-reduce：

```python
dist.init_process_group("nccl", "tcp://localhost:2333", ...)
```

每个 attention layer 的 output projection 和 FFN 的 down projection 之后，各 rank 的输出需要 all-reduce 求和，才能得到完整的 hidden state。这部分通信是 NCCL 在 GPU 间完成的，与 SharedMemory 的"指令广播"是两个完全独立的通道。

---

## 小结

Model Runner 存在的理由是"逻辑-物理翻译"：把 Scheduler 的调度决策（哪些 sequence、处理多少 token）转化成 GPU 可以直接执行的 tensor 和调用接口。

它的初始化顺序有严格逻辑：**先 warmup 量出峰值显存，再按实测结果分配 KV cache，最后 capture CUDA graph**——任何一步的结果都是下一步的前提。

Prefill 和 decode 的数据格式天壤之别：prefill 是变长 batch，Flash Attention 需要 `cu_seqlens`；decode 是定长 batch，每 sequence 一个 token，Flash Attention 需要 `block_tables` 和 `context_lens`。Context 对象作为进程级单例，让 attention layer 在 forward 时能拿到这些"推理步配置"，而无需污染模型接口。

CUDA Graph 把 decode 阶段的 Python dispatch overhead 从热路径里消除，代价是需要为每个 bucket size 各 capture 一份 graph，并在 replay 前原地更新输入 tensor。

多 GPU 下，rank 0 通过 SharedMemory + Event 广播调度指令，各 rank 执行完全相同的 forward（但权重分片不同），NCCL 负责 forward 过程中的中间激活 all-reduce。两套通信机制各司其职：一个传 Python 对象，一个传 GPU tensor。
