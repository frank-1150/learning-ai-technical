---
date: 2026-07-24
title: "How Much Memory and Compute Does Training a GPT-2 XL Take? — Doing the AdamW Accounting"
description: Using CS336 Assignment 1's GPT-2 XL as the worked example, we break down training memory (params/gradients/optimizer state/activations) and FLOPs line by line, derive the 16 bytes/param and 6ND rules of thumb, and answer "a single GPU only fits batch=5, so how do you train at batch=1024?"
tags: [cs336, adamw, memory, flops, gpt-2, training, gradient-accumulation, data-parallel]
---

# How Much Memory and Compute Does Training a GPT-2 XL Take?

> These are my notes on one problem from Chapter 4 (*Training LLM*) of CS336 Assignment 1. The question itself is plain: given GPT-2 XL's hyperparameters, write down peak memory and FLOPs as a function of batch size. (Before it you implement AdamW by hand, which is what makes the memory analysis tractable.)
>
> I think it's worth writing up on its own, because working through it tells you what a single GPU is actually doing during training, why it takes so long, why multi-GPU is not optional, and what tricks multi-GPU requires. It's a complete resource-accounting exercise for a GPT-scale training run.

## Model configuration

We use a GPT-2 XL sized model. (The CS336 implementation uses RMSNorm + SwiGLU, slightly different from the original GPT-2's LayerNorm + GELU.)

| Symbol | Meaning | Value |
|---|---|---|
| $V$ | Vocabulary size | 50,257 |
| $T$ | Context length | 1,024 |
| $L$ | Transformer layers | 48 |
| $D$ | Model dimension `d_model` | 1,600 |
| $H$ | Attention heads | 25 |
| $d_{ff}$ | FFN inner dimension | 4,288 |

::: tip Why d_ff = 4288 and not 4×1600?
SwiGLU has **three** weight matrices ($W_1, W_2, W_3$) instead of a vanilla FFN's two. To keep the parameter count level with a 4× vanilla FFN, the convention is $d_{ff} \approx \frac{8}{3} D$, rounded up to a multiple of 64 (for tensor-core alignment): $\frac{8}{3} \times 1600 = 4266.7 \rightarrow 4288$.
:::

## 1. Parameter count: ~1.64 billion

Per Transformer block:

$$
\underbrace{4D^2}_{Q,K,V,O} + \underbrace{3 \cdot d_{ff} \cdot D}_{\text{SwiGLU}} + \underbrace{2D}_{\text{two RMSNorms}} = 30{,}825{,}600
$$

| Component | Parameters |
|---|---|
| One block | 30,825,600 |
| × 48 layers | 1,479,628,800 |
| Embedding + LM Head + final RMSNorm | 160,824,000 |
| **Total** | **1,640,452,800** |

Note that last row: the embedding and the LM head are $VD = 80$M parameters each, 160M together — **10% of the total** — yet each participates in exactly one matmul. This is why many models use **weight tying** (sharing the embedding and LM head weights): it instantly saves 5% of the parameters, gradients, and optimizer state.

## 2. Memory: the magic number is 16 bytes/param

Assume float32 (4 bytes) throughout. Training memory splits into four buckets:

| Category | Expression | Value |
|---|---|---|
| Parameters | $4 \times P$ | 6.56 GB |
| Gradients | $4 \times P$ | 6.56 GB |
| Optimizer state ($m$ and $v$) | $8 \times P$ | 13.12 GB |
| **Subtotal (independent of batch size)** | $\mathbf{16 \times P}$ | **26.25 GB** |
| Activations | $4L(7BTD + BHT^2 + 3BT \cdot d_{ff})$ | **9.76·B GB** |

::: important Remember this: fp32 + AdamW = 16 bytes/param
Parameters 4 + gradients 4 + AdamW's first moment $m$ 4 + second moment $v$ 4 = **16 bytes**.

In other words, **before the model computes anything at all, memory is already 16× the parameter count**. A 1.64B model costs 26 GB just to sit there. This is why you cannot full-finetune a 7B model on a 24GB 4090 (7×16 = 112 GB), while inference on it needs only 14 GB (fp16).

With plain SGD (no momentum) the number is 8 bytes/param; SGD with momentum is 12. AdamW's convergence advantage is bought with an extra 2× the parameter count in memory.
:::

### Where the activation term comes from

Intermediate results each layer must keep for backward (counted in floats):

- $7BTD$ — RMSNorm inputs, $Q$/$K$/$V$, attention output, residuals, and various $B \times T \times D$ tensors around the FFN
- $BHT^2$ — the attention score matrix $QK^\top$, shape $B \times H \times T \times T$
- $3BT \cdot d_{ff}$ — the three SwiGLU branch activations

At $B=1$: 50,855,936 floats per layer, 2.44B across 48 layers, × 4 bytes = **9.76 GB**.

::: warning This is a rough lower bound
Implementations differ a lot in how many intermediates they keep (fused kernels, whether softmax output is saved, dropout masks, etc.), so the coefficient $7BTD$ is an order-of-magnitude estimate, not an exact count — real PyTorch implementations are usually larger. What matters is the **structure**: activations scale linearly with $B$, and the $BHT^2$ term scales **quadratically** with sequence length.

That $BHT^2$ term is 26.2M at $T=1024$, roughly half the per-layer total. Push $T$ to 8192 and it grows 64×, while the other terms grow only 8× — **which is exactly why FlashAttention exists**: tiling keeps the $N \times N$ score matrix from ever landing in HBM, turning this term from $O(T^2)$ into $O(T)$ in memory.
:::

### Final expression

$$
\text{Peak Memory} \approx 26.25 + 9.76 \cdot B \quad \text{(GB)}
$$

## 3. What batch size fits on an 80GB H100?

**Training** (with AdamW state):

$$
26.25 + 9.76B \le 80 \implies B \le 5.50 \implies B_{max} = 5
$$

**Inference** (parameters only):

$$
6.56 + 9.76B \le 80 \implies B \le 7.52 \implies B_{max} = 7
$$

::: note That inference number badly understates reality
Inference has no backward pass, so it **keeps no intermediate activations** — once a layer is done its intermediates can be dropped, leaving only a single layer's working buffer.

And precisely because layers pass forward a single $B \times T \times D$ tensor and never need to come back, inference is a natural fit for **pipeline parallelism**: split the 48 layers into segments across GPUs and hand the activation down the line. Communication is tiny compared with tensor parallelism, which needs an all-reduce every layer. Pipelining is awkward in *training* because backward has to walk back through the stages, creating bubbles; inference has no such problem — which is why multi-node LLM serving typically looks like "tensor parallel within a node, pipeline parallel across nodes."

So the real memory bottleneck at inference isn't activations, it's the **KV cache**:

$$
\text{KV Cache} = 2 \times L \times B \times T \times D \times \text{bytes} = 2 \times 48 \times 1024 \times 1600 \times 2\,\text{(fp16)} = 0.31\,\text{GB} \cdot B
$$

By that math an 80GB card serves up to $B \approx 230$. That 30× gap is exactly what vLLM / PagedAttention exist to manage carefully.
:::

## 4. FLOPs: where the 6ND rule comes from

### Forward pass through one Transformer block (B=1)

| Operation | FLOPs | Value |
|---|---|---|
| Q, K, V projections (3×) | $6BTD^2$ | 15.73 GFLOPs |
| Attention scores $QK^\top$ | $2BT^2D$ | 3.36 GFLOPs |
| Attention weighted sum $PV$ | $2BT^2D$ | 3.36 GFLOPs |
| Output projection | $2BTD^2$ | 5.24 GFLOPs |
| SwiGLU ($W_1, W_2, W_3$) | $6BTD \cdot d_{ff}$ | 42.15 GFLOPs |
| **Per-layer total** | | **69.84 GFLOPs** |

::: tip One matmul = 2mnk
Every factor of $2$ above traces to the same fact: multiplying $m \times k$ by $k \times n$ takes $mnk$ multiplies and $mnk$ adds, i.e. $2mnk$ FLOPs. Given the shapes and this relation, you can derive every row above yourself.
:::

Worth noticing: **SwiGLU is 60% of a layer** (42.15 / 69.84), while the two $T^2$ attention terms together are only 10%. At $T = 1024$, a Transformer's compute mostly goes to the FFN; "attention is the bottleneck" only becomes true at long context (the $T^2D$ terms overtake the $TD^2$ terms once $T > D$, i.e. $T = 1600$ here).

### Total FLOPs per training step (B=1)

| Phase | FLOPs | TFLOPs |
|---|---|---|
| Forward (48 layers + LM head) | 3,516,769,894,400 | 3.52 |
| Backward (≈ 2× forward) | 7,033,539,788,800 | 7.03 |
| **Forward + backward** | **10,550,309,683,200** | **10.55** |
| AdamW update | 24,606,792,000 | 0.025 |

::: note Why backward ≈ 2× forward
For $Y = XW$, forward is one matmul. Backward computes **two** gradients: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top$ (passed upstream) and $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$ (used to update weights). Two matmuls, same shapes as forward — hence 2×.
:::

The AdamW update (~15 element-wise ops per parameter) is **0.23%** of forward+backward — negligible.

But note: negligible in FLOPs does not mean negligible in time, because this step is purely **memory-bound**. "Memory-bound" means the bottleneck is memory bandwidth rather than compute — the GPU's ALUs sit idle waiting for data. Updating each parameter touches four tensors:

| Operation | Tensors | Traffic |
|---|---|---|
| Read | params $\theta$, gradient $g$, moment $m$, variance $v$ | $4P$ |
| Write | params $\theta$, moment $m$, variance $v$ | $3P$ |
| **Total** | | **$7P$ element accesses ≈ 4 parameter-sized tensors round-tripped** |

In fp32 with $P = 1.64\text{B}$, that's $7 \times 1.64\text{B} \times 4\,\text{bytes} \approx 46\ \text{GB}$ of memory traffic for only 24.6 GFLOPs of arithmetic. That's an **arithmetic intensity of ~0.5 FLOP/byte**, where an H100 needs roughly 300 FLOP/byte to saturate its compute — nearly three orders of magnitude short. At the H100's 3.35 TB/s, just moving those 46 GB takes ~14 ms; the 0.025 TFLOPs of math would take 0.1 ms.

So in a real profile, the optimizer step's share of wall-clock time is far above 0.23%. That's exactly what **fused optimizer kernels** are for — collapsing the element-wise ops into a single kernel that reads and writes memory once.

### Cross-checking against the 6ND rule

$$
\text{FLOPs per step} \approx 10.55 \cdot B \ \text{(TFLOPs)}
$$

The common rule of thumb is **total training FLOPs ≈ 6ND** ($N$ = parameters, $D$ = tokens). Check it: at $B=1$ a step processes 1024 tokens, so

$$
6 \times 1.64 \times 10^9 \times 1024 = 1.01 \times 10^{13} = 10.1\ \text{TFLOPs}
$$

Within 4% of our line-by-line 10.55. The excess is precisely the attention $T^2$ terms — **6ND ignores attention by construction, which holds when $T \ll D$ and underestimates at long context**.

## 5. How long does one H100 need?

| Parameter | Value |
|---|---|
| H100 peak (TF32) | 495 TFLOP/s |
| MFU (Model FLOPs Utilization) | 50% |
| Effective throughput | 247.5 TFLOP/s |
| Batch size | 1,024 |
| Training steps | 400,000 |

$$
\begin{aligned}
\text{FLOPs per step} &= 10.55 \times 1024 = 10{,}804\ \text{TFLOPs} \\
\text{Total FLOPs} &= 400{,}000 \times 10{,}804 = 4.32 \times 10^{21} \\
\text{Time} &= \frac{4.32 \times 10^{21}}{247.5 \times 10^{12}} = 1.75 \times 10^7\ \text{s} \\
&\approx 4{,}850\ \text{hours} \approx \mathbf{202\ days} \approx 6.7\ \text{months}
\end{aligned}
$$

That processes $1024 \times 1024 \times 400{,}000 \approx$ **419B tokens** in total.

::: warning About that 495 TFLOP/s
For H100 SXM TF32 tensor cores, ~495 TFLOP/s is the **dense** figure; the larger number on the spec sheet assumes 2:4 structured sparsity. Real LLM training doesn't use sparsity, so dense is the right basis. Also, real training almost always uses BF16 (~990 TFLOP/s), which is twice as fast.

**MFU (Model FLOPs Utilization)** is the fraction of peak actually achieved. 50% is fairly optimistic — GPT-3-scale runs typically land at 35–45%, and small models or pipelines with many bubbles can fall below 20%.
:::

::: tip Is 419B tokens sensible?
Under the **Chinchilla** scaling law ($D \approx 20N$), the compute-optimal token budget for 1.64B parameters is about **33B tokens**. 419B is 12.7× that — heavily **over-trained**.

Modern models (Llama 3, Qwen 3, …) all blow past the Chinchilla ratio, because Chinchilla optimizes "best loss for a given training budget" while industry actually cares about **inference cost**: a small over-trained model is cheaper to serve forever than a large under-trained one. Train once, infer a billion times.
:::

## 6. One GPU fits B=5 — so how do you get B=1024?

This is the glaring contradiction between sections 3 and 5: we computed a max of 5 per card, then assumed a batch size of 1024.

### Option 1: Gradient accumulation

Trade time for space. Gradients are **linearly additive over the batch**, so you can run several forward+backward passes, accumulate, and update once:

```
effective_batch     = 1024
micro_batch         = 5
accumulation_steps  = ⌈1024 / 5⌉ = 205
```

Each training step:

1. Loop 205 times: forward(micro_batch=5) → loss → backward (gradients **accumulate**, no zeroing)
2. After accumulating, call `optimizer.step()` once
3. `optimizer.zero_grad()`

Mathematically identical to processing $B=1024$ in one go. **Total FLOPs are unchanged**, so a single card is still those 202 days — gradient accumulation solves "doesn't fit," not "too slow."

::: warning A common pitfall
If your loss uses `mean` reduction, divide each micro-batch's loss by `accumulation_steps` before backward, or the accumulated gradient will be 205× too large. Also 205 × 5 = 1025 ≠ 1024; in practice you pick a micro_batch that divides evenly (e.g. 4, for 256 steps).
:::

### Option 2: Data parallelism

Every GPU holds a **full model replica** and processes a **different data shard**: each does forward + backward → all-reduce to average gradients → each runs `optimizer.step()`.

| GPUs | Per-GPU batch size | Estimated training time |
|---|---|---|
| 1 | 5 (+ grad accumulation) | ~202 days |
| 8 | 128 | ~25 days |
| 32 | 32 | ~6 days |
| 64 | 16 | ~3 days |

Note this table assumes **ideal linear scaling**. In reality every step all-reduces all 1.64B gradients (6.56 GB in fp32), and communication's share grows with GPU count. Within a node NVLink (~900 GB/s) copes; across nodes InfiniBand becomes the bottleneck — which is why [NVSwitch / NVL72 interconnect topologies](../inference/nvlink-nvswitch-topology) exist.

Also note: **pure data parallelism saves no memory** — every card still carries the full 26.25 GB of static overhead. Which leads to the next two options.

### Option 3: ZeRO / FSDP — shard the static overhead too

Of the 16 bytes/param, 12 are gradients and optimizer state. So shard those across GPUs as well:

| Stage | Sharded | Per-GPU memory with N GPUs |
|---|---|---|
| ZeRO-1 | Optimizer state | $4P + 4P + 8P/N$ |
| ZeRO-2 | + gradients | $4P + (4P + 8P)/N$ |
| ZeRO-3 / FSDP | + parameters | $16P/N$ |

With ZeRO-3 on 8 GPUs, static overhead drops from 26.25 GB to 3.3 GB per card. The cost is all-gathering parameters back on demand, which raises communication volume substantially.

### Option 4: Activation checkpointing — trade the activations too

That 9.76·B GB can also be traded for time: keep only each layer's input (rather than every activation) and recompute the intermediates with an extra forward pass during backward. Memory drops from $O(L)$ to $O(\sqrt{L})$ or even $O(1)$, at the cost of one more forward — total FLOPs go from 3× to 4× (roughly +33% time).

This is mandatory for very-long-context training.

### Option 5: Model parallelism — when the model itself doesn't fit

| Strategy | Approach |
|---|---|
| **Tensor parallelism** | Split a single matmul across GPUs (e.g. shard the FFN along $d_{ff}$). Requires communication every layer, so normally confined to NVLink within a node. |
| **Pipeline parallelism** | Put different layers on different GPUs and pipeline through them. Low communication volume, but introduces bubbles. |

::: note
Real large-scale training **combines** all of these — so-called 3D/4D parallelism: tensor parallel (intra-node) × pipeline parallel (inter-node) × data parallel (across replicas), plus ZeRO, gradient accumulation, and activation checkpointing. That's the subject of the distributed-training half of CS336.
:::

## 7. What to take away

If you remember only three things:

1. **16 bytes/param** — static training memory for fp32 + AdamW. To judge whether a model is trainable at all, multiply the parameter count by 16.
2. **6ND** — total training FLOPs. To estimate training cost: params × tokens × 6, divided by (peak FLOP/s × MFU).
3. **Activations scale with $B$; static overhead doesn't** — so batch size only gets whatever memory is left over, and "left over" is usually painfully little.

Together these three let you answer, on the back of a napkin, "how many GPUs, how many days, how many dollars to train an X-billion model" — which happens to be the question the AI industry asks most often and actually computes least often.
