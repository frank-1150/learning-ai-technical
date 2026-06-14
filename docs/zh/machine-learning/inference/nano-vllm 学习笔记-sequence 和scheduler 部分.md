---
date: 2026-06-14
title: "nano-vllm 学习笔记（二）：Sequence 状态机与 Scheduler 调度设计"
description: 从一个请求进来到最终输出，nano-vllm 里 Sequence 对象是怎么设计的、跨进程传递如何工作、Scheduler 为什么优先 prefill、chunked prefill 为什么只对第一条序列开放——逐一讲清楚。
tags: [inference, vllm, scheduler, sequence, tensor-parallel, prefill, decode, nano-vllm]
---

# nano-vllm 学习笔记（二）：Sequence 状态机与 Scheduler 调度设计

> 上一篇讲了 Block Manager 和 Prefix Cache 的实现。这篇对应 `nanovllm/engine/sequence.py` 和 `nanovllm/engine/scheduler.py`，讲 Sequence 对象是怎么设计的、以及 Scheduler 的调度哲学。

## 一个请求的生命周期

先建立整体感：一个请求从进来到出去经历了什么？

1. 进入 `LLMEngine`，用 tokenizer 把 string 转成 token ID 列表
2. 创建一个 `Sequence` 对象，放进 `waiting` 队列
3. `Scheduler.schedule()` 被调用，从队列里挑 sequence，安排本轮要处理哪些
4. Model Runner 执行推理，生成新 token
5. `Scheduler.postprocess()` 把新 token 追加到 sequence，更新状态
6. 如果 sequence 生成完毕（触发 EOS 或者达到 max_tokens），标记为 FINISHED，释放 block

整个过程的核心状态都存在 `Sequence` 对象里。

## Sequence：为什么需要这个对象

先问一个问题：为什么不直接用 dict 或 tuple 记录请求的状态，而要单独定义一个 `Sequence` 类？

因为一个请求在 inference 过程中不是静止的——它会从 WAITING 变成 RUNNING，再变成 FINISHED，中途可能被 preempt 打回 WAITING，block_table 也在动态变化。这是一个有**状态跃迁**的对象，自然应该用类来封装，让状态机的转换逻辑集中在一个地方而不是散落在 scheduler 各处。

## Sequence：一个请求在 token 视角下的快照

```python
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids, sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

几个不那么直白的属性说一下：

**`num_cached_tokens` vs `num_scheduled_tokens`**

- `num_cached_tokens`：KV cache 里已经有的 token 数量。这些 token 对应的 K/V 不需要重新计算，可以直接从显存里读
- `num_scheduled_tokens`：本轮推理要处理的 token 数量。对于 prefill 来说，这是新进来的一批 token；对于 decode 来说，通常是 1

两者相加如果等于 `num_tokens`，说明这轮 prefill 就可以完结，sequence 进入 decode 阶段。

**`is_prefill`**

初始化为 True，因为任何 sequence 进来都得先做 prefill。完整 prefill 完成之后会设为 False，进入 decode。另外，如果 sequence 因为显存不足被 **preempt（抢占）**，需要重新放回 waiting 队列重新 prefill，`is_prefill` 也会被重置为 True。

**`ignore_eos`**

EOS（end of sequence）是模型输出的一个特殊 token，表示生成结束。正常情况下，生成到 EOS 就停止。但有些场景不想这样——比如你在做评测，想让模型固定生成 max_tokens 个 token 而不是遇到 EOS 就停；或者在生成包含特殊字符的内容时，模型可能会错误地提前 EOS。`ignore_eos=True` 就是告诉 scheduler：遇到 EOS 也不要停，继续生成直到 max_tokens。

**`block_table`**

记录这个 sequence 对应的 block ID 列表，顺序和 token 序列对应。Block 0 存 tokens[0:256]，Block 1 存 tokens[256:512]，以此类推。Scheduler 根据这个列表告诉 model runner 去哪里读 KV cache。

### 状态机

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

- **WAITING**：在等待队列里，还没开始 prefill（或者被 preempt 之后重新等待）
- **RUNNING**：prefill 已完成，正在 decode
- **FINISHED**：生成结束，等待清理

### `__getstate__` / `__setstate__`：为什么要自定义序列化

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
        self.num_scheduled_tokens, self.block_table, last_state = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```

这里用到了 Python pickle 的自定义序列化接口。`__getstate__` 定义"序列化时打包哪些字段"，`__setstate__` 定义"反序列化时怎么还原对象"。

**为什么要手动序列化？**

nano-vllm 使用 Tensor Parallel 时，主进程负责 scheduler（决定哪些 sequence 进入本轮 batch、分配 KV cache block），worker 进程负责实际的模型 forward。每一步推理前，主进程需要把调度结果通过 IPC 传给 worker：哪些 sequence 参与本轮、它们的 token 状态、`block_table`（logical block id → physical KV cache block id 的映射，worker forward 做 attention 时需要知道去哪几个 KV block 读历史 K/V）、以及这轮是 prefill 还是 decode。

如果直接把 Python `Sequence` 对象传过去，Python 多进程会默认走 pickle 序列化，对象越复杂开销越大，还可能遇到不可序列化字段的问题。所以 nano-vllm 在 `__getstate__` 里手动把 sequence 压成一个轻量 tuple，只打包 worker 真正需要的字段，让 IPC 更轻、更可控。

这里值得多问一句：**为什么要用多进程，而不是用多线程来管理多块 GPU？** 原因不只是 GIL。PyTorch/CUDA 的重型计算发生在 C++/CUDA kernel 里，很多时候会释放 GIL，所以多线程并非完全不能并行。真正的原因是 TP 的执行模型本身：TP 以 **rank** 为单位组织执行——每个 rank 对应一个 GPU，拥有自己的模型分片、CUDA context、KV cache 视图和 NCCL 通信状态。让每个 rank 跑在一个独立进程里，比在一个 Python 进程内用多线程管理多个 rank 的状态要简单得多，也更符合 PyTorch distributed / NCCL 的使用惯例。另外，CUDA 多进程需要注意启动方式：不能用 `fork` 后再初始化 CUDA（会遇到 "Cannot re-initialize CUDA in forked subprocess" 报错），通常必须用 `spawn`。

**prefill 和 decode 传的信息量不同**

关键的优化在这里：

- **prefill 阶段**：需要把完整的 `token_ids` 传过去，因为 prefill 要对所有 token 计算 K/V，worker 进程需要知道完整的输入
- **decode 阶段**：`is_prefill` 已经是 False，只需要传 `last_token`（最新生成的那个 token）。原因是 decode 只需要对最新一个 token 计算 query 向量，然后和存好的 KV cache 做 attention——之前所有 token 的 K/V 都已经在 GPU 显存里了，不需要重新传

这个优化大幅减少了进程间通信量：一个 8192 token 的长序列在 decode 阶段，每步只需要传 1 个 int 而不是 8192 个。

---

## Scheduler：调度的哲学

Scheduler 的核心职责是：**在每个 step，从 waiting 和 running 队列里决定这一轮处理哪些 sequence**。

初始化的几个关键配置：

```python
def __init__(self, config: Config):
    self.max_num_seqs = config.max_num_seqs             # 单 batch 最多几个 sequence
    self.max_num_batched_tokens = config.max_num_batched_tokens  # 单 batch 最多几个 token
    self.block_manager = BlockManager(...)
    self.waiting: deque[Sequence] = deque()
    self.running: deque[Sequence] = deque()
```

`max_num_seqs` 限制并发数，`max_num_batched_tokens` 限制单轮计算量（防止 OOM）。

### prefill 优先，decode 靠后

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # 先跑 prefill
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        ...

    if scheduled_seqs:
        return scheduled_seqs, True   # True 表示这轮是 prefill

    # waiting 队列空了才跑 decode
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        ...
    return scheduled_seqs, False
```

**为什么 prefill 完全优先于 decode？**

这里有几个原因：

1. **TTFT（Time To First Token）是用户最感知的延迟**。prefill 完成之前，用户看不到任何输出。优先跑 prefill，就是优先让排队的用户"开始看到响应"，而不是让已经开始 decode 的请求更快结束。

2. **prefill 和 decode 的计算特征完全不同，混批处理会带来对齐问题**。prefill 是"一次处理大量 token"（compute-bound），decode 是"每次只处理一个 token"（memory-bound）。把两者放在同一个 batch 里，实现起来复杂，且 decode 的单 token 会"拖慢" prefill 的吞吐。nano-vllm 选择彻底分离，每个 step 要么全是 prefill，要么全是 decode，逻辑简单且易于优化。

3. **waiting 非空意味着系统已经"欠账"**。已经在 decode 的请求可以等一步，让新进来的请求先开始响应，从整体排队体验上更公平。

注意 vLLM 的生产版本支持 **chunked prefill 与 decode 混批**，可以兼顾 TTFT 和吞吐。nano-vllm 的"prefill 完全优先"是一个简化策略。

### Prefill 的调度细节

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]                    # 只看队头
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break                            # 显存不足，停
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs:  # ← 关键条件
        break
    ...
```

**chunked prefill 只对第一条序列开放**

`if remaining < num_tokens and scheduled_seqs: break` 这一行的含义是：如果这个 sequence 的 token 数超过了本轮还剩的 quota，**而且已经有其他 sequence 被安排了**，就跳过它。

反过来说：如果它是本轮**第一个** sequence（`scheduled_seqs` 为空），即使它超出了 quota，也不会 break——会继续执行，用 `min(num_tokens, remaining)` 截断，只做部分 prefill（chunked prefill）。

**为什么只允许第一条做 chunked prefill？**

这是一个工程简化，但有合理的动机：

如果允许中间的某条 sequence 被截断，就会出现 "prefill batch 里一部分 sequence 是完整的、一部分是截断的" 的情况。截断的 sequence 这轮没做完 prefill，下一轮还得继续——这意味着 scheduler 需要追踪每个 sequence "做到哪了"，逻辑复杂度大大增加。

而如果只允许**第一条**被截断，那么每轮 batch 里：第一个 sequence 可能是"做了一部分"，之后所有 sequence 要么是"完整一轮 prefill"，要么没被选上。第一条的截断状态直接记在 `num_cached_tokens` 和 `num_scheduled_tokens` 里，下一轮它仍然在 waiting 队头继续。逻辑清晰，状态简单。

**超长 sequence 的处理**

假设 `max_num_batched_tokens = 2000`，进来一条 5000 token 的请求。它会在 waiting 队头，第一次 schedule 时：

- `remaining = 2000`
- `num_tokens = 5000`（假设没有 prefix cache）
- 它是第一个，所以不 break，而是 `num_scheduled_tokens = min(5000, 2000) = 2000`
- 这轮做 2000 token 的 prefill，`num_cached_tokens` 更新到 2000

下一轮它还在 waiting 队头（因为没 prefill 完，status 还是 WAITING），继续做剩下的 3000 token。

**`can_allocate` 返回 -1 就 break**

为什么显存不足时直接 break 而不是跳过这条 sequence 试试下一条？

因为 waiting 队列是严格有序的（FIFO）。如果队头的 sequence 放不下，说明显存已经吃紧，强行跳过去安排后面的 sequence 反而会让队头一直等、造成饥饿（starvation）。更合理的策略是：等显存腾出来（当前 decode 的 sequence 结束了），再来处理 waiting 队列。

### Decode 的调度

```python
# decode
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())    # 抢占 running 队列最后的 seq
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

Decode 阶段每个 sequence 每次只生成 1 个 token，`num_scheduled_tokens = 1`。

这里有一个 **preempt（抢占）机制**：如果显存满了（`can_append` 返回 False），就把 running 队列**最后面**的 sequence 强制释放掉（deallocate 它的 block，状态重置为 WAITING），腾出显存给队头的 sequence 用。被抢占的 sequence 下次会从头重新 prefill。

为什么抢占队尾而不是队头？FIFO 公平原则——队头的 sequence 等待时间最长，优先保证它能继续跑。

### Waiting 队列与 Running 队列的分离

**为什么要两个队列？**

- `waiting` 存放**还没完成 prefill** 的 sequence，包括新进来的请求和被 preempt 的 sequence
- `running` 存放**prefill 已完成、正在 decode** 的 sequence

两者调度策略完全不同：prefill 受 `max_num_batched_tokens` 限制（因为一次处理大量 token，GPU 计算量和显存占用都大）；decode 受 `max_num_seqs` 限制（因为每步只处理 1 个 token，主要瓶颈是能同时维持多少条 KV cache）。

分开存储让 scheduler 的逻辑更清晰，每次 schedule 先处理 waiting（prefill），waiting 空了再处理 running（decode），互不干扰。

### postprocess：推理完成后的收尾

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)       # 把这轮完成的 block 写入 prefix cache
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                               # chunked prefill 还没做完，不追加 token
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

注意 `hash_blocks` 在每次推理后都会被调用——这是 Prefix Cache 的"写入"时机：这轮做完的 block 会被写进 `hash_to_block_id`，下一个有相同前缀的请求就能命中。

Chunked prefill 的情况下，如果 `num_cached_tokens < num_tokens`，说明 prefill 还没做完，这一步不会追加新 token（没有生成新内容），只是更新缓存进度。

## 小结

Sequence 是一个请求在 token 视角下的完整状态快照，它需要在多进程间传递，所以自定义了 pickle 序列化——prefill 传完整 token_ids，decode 只传 last_token。

Scheduler 的核心设计决策是**严格的 prefill 优先**，然后在 prefill 内部，只允许第一条 sequence 做 chunked prefill，保持状态管理的简洁。显存不足时用 preempt 机制应对，优先保护队头、牺牲队尾。

两篇笔记合在一起，就是 nano-vllm 最核心的两层：**物理显存的管理**（Block Manager）和**逻辑调度**（Scheduler）。上面这些搞清楚了，再读 model_runner.py 就会容易得多。
