---
date: "2026-10-05"
meta-description: How prefix caching, FP8, the per-step token budget, and max_num_seqs raised Qwen3.6-35B-A3B vLLM throughput from 3.9 to 20.6 req/s on one H20
title: "5.3× Serving Throughput for Qwen3.6-35B-A3B on a Single H20"
description: A vLLM optimization retrospective grounded in workload shape, roofline analysis, and kernel traces
tags: [vllm, inference, qwen, moe, fp8, prefix-cache, cuda, performance]
---

# 5.3× Serving Throughput for Qwen3.6-35B-A3B on a Single H20: A vLLM Optimization Retrospective

> **TL;DR**: On the same H20 GPU and the same real request-length distribution, the service's max throughput went from 3.9 req/s to 20.6 req/s (×5.3) in four steps: enable prefix caching, switch to FP8 weights, raise the per-step token budget, and raise max_num_seqs. The last three steps all attack the same problem: **each step fed the GPU too few tokens, so the CPU could not launch kernels fast enough to keep the GPU busy**. The final setup reaches 54% of the theoretical roofline.

This post walks through the experiments in order: what we saw at each step, where we concluded the bottleneck was, why that led to the next step, and how to estimate "how far are we from the ceiling". Places marked **🔁 to re-test** are results that did not quite match expectations or whose evidence is not yet solid; the last section collects them into a re-test list.

**Who this is for**: engineers serving MoE or hybrid (linear attention + full attention) models with vLLM who want to understand where throughput is stuck. Basic familiarity with prefill / decode, continuous batching and CUDA kernel launches is assumed.

# Background: model, workload and methodology

## Model and hardware

| Item | Details |
|-|-|
| Model | Qwen3.6-35B-A3B: ~35B total parameters, ~3B active per token. Of its 40 layers, 30 are Gated DeltaNet linear attention (GDN) and 10 are full attention; the FFN is a MoE with 256 experts, top-8 routing, plus one shared expert. |
| Quantization | Baseline: AWQ (W4A16: 4-bit weights, FP16 compute). Optimized: the official FP8 checkpoint (W8A8: linear layers run on FP8 tensor cores). |
| Hardware | Single H20: ~148 TFLOPS dense FP16, ~296 TFLOPS FP8, ~4 TB/s HBM bandwidth. |
| Inference engine | vLLM 0.25.1 with chunked prefill and torch.compile + CUDA graphs on by default (CUDA graphs only cover steps of ≤256 tokens). |

## Workload

The task is "read a long document, write a short JSON": inputs are ~3,200 tokens (p50 ~3,150), of which the first ~2,570 tokens are a system prompt identical across all requests; outputs are short, p50 ~73 tokens. This shape drives almost every decision below:

- **Prefill dominates**: per request, prefill compute far exceeds decode compute.
- **Long shared prefix**: without reuse we recompute the same text over and over.
- **Short decode**: a request stays in the batch for only ~73 steps, so the scheduler must keep admitting new requests.

## How we measured

Every step was re-measured with the same home-grown open-loop load generator so results are comparable:

- **Request content**: token IDs are random (no real data), but (input length, output length) pairs are sampled from real requests; every request shares the same fixed 2,570-token prefix, reproducing the real prefix-hit rate (66% measured, matching production). Output length is fixed with `ignore_eos` + `min_tokens`.
- **Arrivals**: Poisson arrivals rather than a fixed-concurrency closed loop. As we will see, this difference changes which limit is the bottleneck (max_num_batched_tokens vs max_num_seqs).
- **Max throughput**: 60 s of overload at 30 or 60 req/s, mean of 3 repeats; TTFT / TPOT recorded at moderate load (6 req/s).
- **Warm-up**: 3 warm-up rounds after every restart before counting. We learned this the hard way; see "Pitfalls".
- **Profiler**: kernel-level analysis uses vLLM's built-in torch profiler in separate runs; throughput numbers only come from runs without the profiler.

# Overview: four steps to 5.3×

![Figure 1: Measured max throughput per step (bars) vs three theoretical ceilings (lines). Orange = AWQ, blue = FP8](/machine-learning/inference/qwen36-vllm-throughput/en/fig01_roadmap.png)

| Step | Max throughput | Share of roofline | Key observation |
|-|-|-|-|
| **S0 baseline**  <br/>AWQ, no prefix cache, budget 2048, seqs 128 | 3.9 req/s | 46% (ceiling 8.5) | ~2/3 of FLOPs go to recomputing the identical system prompt; the logs show a KV cache hit rate of 0. |
| **S1 + prefix cache** | 9.9 req/s (×2.5) | 45% (ceiling 21.9) | 66% prefix hit; with AWQ the GPU is nearly always busy, i.e. compute-bound. |
| **S2 + FP8 weights** | 12.0 req/s (×3.1) | 35% (ceiling 33.7) | Kernels got 1.7× faster but end to end only +21%: the GPU spends most of its time waiting for the CPU. |
| **S3 + larger budget**  <br/>max-num-batched-tokens 2048 → 16384 | 15.9 req/s (×4.1) | 47% (ceiling 33.7) | More tokens per step, but under real load capped by max_num_seqs at ~1,750. |
| **S4 + larger max_num_seqs**  <br/>128 → 384 | 20.6 req/s (×5.3) | 54% (ceiling 38.2) | ~10,000 tokens per step; GPU idle drops to 1.7%. |

The figure below strings together the "what we saw → what we did" chain; each following section expands one step.

![Figure 2: The decision chain](/machine-learning/inference/qwen36-vllm-throughput/en/fig02_decision_chain.png)

# Step 1: enable prefix caching (3.9 → 9.9 req/s)

## Why this first

At baseline each request prefills ~3,200 tokens, 2,570 of which are the identical system prompt. By FLOP count, a request costs ~16.8 TFLOP without reuse and ~6.1 TFLOP if the shared prefix's KV and state can be reused. That is a change that **cuts 2/3 of the compute with no accuracy loss** and raises the theoretical ceiling from 8.5 to 21.9 req/s, so it goes first.

## The hybrid-architecture (GDN + attention) catch: the block size is 1056, not 16

For a plain Transformer, prefix caching is just `--enable-prefix-caching`. This model has 30 GDN linear-attention layers:

What Gated DeltaNet caches is a recurrent state $S_t$, updated at every step from the current token. The state has a fixed size of $d \times d$.

GDN has no per-token KV. Instead each sequence keeps a fixed-size recurrent state (~2.1 MB per layer, FP32). To cache "the result of the prefix up to position p" you have to store a snapshot of that state at p. vLLM handles it like this:

- Full-attention KV pages and GDN state pages share one page size, so an attention page must be large enough to hold one GDN state. **That stretches the block size to 1,056 tokens.**
- vLLM must be started with `--mamba-cache-mode align`: state is only saved at block boundaries, and prefill chunks (the chunked-prefill chunk size) are forced to multiples of 1,056.

As a result the prefix can only hit in whole blocks: the 2,570-token shared prefix hits only the first 2 blocks (2,112 tokens), and the remaining 458 tokens are recomputed every time. The hit rate is 66% (2,112 / 3,200), matching production.

> **A free optimization left on the table**: if the shared part of the system prompt were sized to a multiple of 1,056 (e.g. trimmed to 2,112 tokens, or padded with stable content to 3,168), the whole prefix would hit. This post did not run that experiment.

## Results

- Max throughput 3.9 → 9.9 req/s (×2.54).
- Replaying real requests: end-to-end latency p50 2,982 → 1,545 ms, time to first token p50 1,449 → 699 ms.
- Cost: the larger blocks reduce usable KV capacity by ~7.9%.
- Quality: an offline comparison on 100 real samples across 4 evaluation dimensions showed 3 dimensions within noise and 1 lower but not significant after multiple-comparison correction.

> **Re-tested since**: we later re-ran the comparison on two prompts with 500 production samples each and a "same-config rerun" as a noise reference; AWQ + prefix cache overlaps the rerun on every dimension. See "Quality check".

# Step 2: switch to FP8 weights (9.9 → 12.0 req/s)

## Why this

With prefix caching on, the AWQ GPU was busy almost all the time (~2% idle between kernels): clearly compute-bound. Most of the compute is in linear layers (attention projections + MoE experts), and H20's FP8 throughput is 2× its FP16 throughput. AWQ stores 4-bit weights but computes in FP16; with W8A8 FP8, the linear layers can use FP8 tensor cores. A FLOP-based breakdown predicted roughly a 1.8× speed-up.

## Where W8A8 actually quantizes: precision flow through one forward pass

"W8A8" means both the weights (W) and the activations (A) of linear layers are 8-bit floats (FP8 E4M3, max ≈ ±448); everything else stays in BF16 / FP32. The [official FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) uses block-wise scaling: weights per 128×128 block and activations per "128 channels of one token", each group with its own FP32 scale (`weight_block_size: [128, 128]`, `activation_scheme: dynamic` in the config).

![Figure 2b: Precision flow of W8A8 inference. Top: inside one FP8 linear layer; bottom: the precision of each step in one decoder layer](/machine-learning/inference/qwen36-vllm-throughput/en/fig02b_w8a8_precision_flow.png)

**How one FP8 linear layer (Y = X · W) is computed**:

1. **Weights: quantized offline, no conversion at runtime.** At export time each 128×128 block is divided by its scale `s_w = max|W_block| / 448` and stored as FP8; `s_w` itself is stored in FP32.
2. **Input X: quantized on the fly in every forward pass (BF16 → FP8, down-cast).** For every token and every 128 channels, take the max absolute value, set `s_x = max / 448`, and round `X / s_x` to FP8. This is a separate small kernel, the ~5% "activation quantization" cost seen in profiling.
3. **Matmul: FP8 × FP8 on tensor cores**, 2× the FP16 throughput on H20.
4. **Accumulation: FP32 (up-cast).** After each 128-wide K block, the partial sum is multiplied by `s_x · s_w` to restore its magnitude and added into an FP32 accumulator.
5. **Output: FP32 → BF16 (down-cast)**, handed to the next op.

Scaling per group of 128 matters because E4M3 has only 3 mantissa bits: with a single scale for the whole matrix, one outlier would flush everything else to zero; with groups, the error stays local.

| Category | What it covers in Qwen3.6-35B-A3B |
|-|-|
| **Quantized to FP8** (weights + activations) | Full-attention q / k / v / o projections; GDN in_proj_qkvz and out_proj; gate_up / down of all 256 MoE experts and the shared expert. That is, most parameters and most FLOPs. |
| **Not quantized** (stays BF16) | Embeddings, lm_head, all RMSNorms, the router gate, the shared-expert gate, GDN in_proj_a / b, conv1d, A_log, dt_bias; the attention computation itself, activation functions, residual adds; the KV cache (`kv_cache_dtype=auto`, i.e. BF16). |
| **Up-cast** (to FP32) | Accumulation inside every FP8 matmul; RMSNorm internals; attention softmax and accumulation; router softmax / top-8; the GDN recurrent state (`mamba_ssm_dtype: float32`, which is also the state snapshot stored by the prefix cache); logits before sampling. |
| **Down-cast** (lower precision) | Activation quantization BF16 → FP8 before every FP8 linear layer, several times per layer (2 per GDN layer, 2 per attention layer, 2 for every selected expert and the shared expert); every matmul output FP32 → BF16. |

This also explains two things we see later: only the matmuls get FP8's 2× compute, while attention and GDN compute is unchanged; and every FP8 linear layer adds a small quantization kernel, so when steps are small these extra launches amplify the CPU-side bottleneck. Also note that the prefix cache stores BF16 KV and FP32 GDN state, not FP8.

## Below expectations: only 21% faster

Throughput only went from 9.9 to 12.0 req/s. Profiling the kernel timeline made the reason clear:

- **GPU compute itself delivered**: total FP8 kernel time is 41% lower than AWQ (1.70×); MoE is 2.3× faster, dense GEMMs 1.7× faster; attention and GDN are still BF16 and unchanged; the new activation-quantization kernels take ~5%.
- **But the GPU computes only ~40% of the time** (AWQ ~90%). Before each FP8 kernel the GPU idles ~42 µs on average (AWQ ~6 µs), and the idle time is almost entirely many sub-millisecond gaps. This is classic **launch-bound** behavior: the GPU finishes a kernel and the CPU has not launched the next one yet.

> **The profiling traces**
>
> ![Perfetto trace of the FP8 model, showing frequent gaps between kernels on the GPU stream](/machine-learning/inference/qwen36-vllm-throughput/shared/trace_fp8_perfetto.png)
>
> FP8 profile.
>
> ![Perfetto trace of the AWQ INT4 model, showing a more continuous GPU stream](/machine-learning/inference/qwen36-vllm-throughput/shared/trace_awq_perfetto.png)
>
> INT4 (AWQ) profile. The FP8 GPU stream clearly has many more gaps than INT4.
>
> **How to open a trace**
>
> 1. Recommended: Perfetto. Open https://ui.perfetto.dev, click "Open trace file" on the left and pick the `.json.gz` directly, no need to unzip. The file is parsed locally in the browser and not uploaded. Unzipped it is ~300 MB, so the first load takes tens of seconds.
> 2. `chrome://tracing` also works but is older and sluggish on large files.
> 3. TensorBoard's PyTorch Profiler plugin works too, but needs extra dependencies; not recommended.
>
> **How to see "the GPU is waiting for the CPU" in Perfetto**
>
> 1. Find the GPU stream row (all kernels live there) and the CPU main-thread row (where the `execute_context_...` annotations are).
> 2. Select one `execute_context_1(1056)_generation_N` annotation and press `F` to zoom to that step.
> 3. In the FP8 @2048 trace, the GPU row has many tiny gaps between kernels. Click any kernel: an arrow links back to the matching `cudaLaunchKernel` on the CPU, and you can see the GPU runs each kernel "as soon as it is launched", then waits for the next.
> 4. Look at the same place in the `fp8_b4096` trace: the GPU row is essentially solid and the CPU's launches run well ahead of the GPU.

- **Why FP8 is more prone to being launch-bound, two reasons**:

  1. FP8 needs less time for the same compute, so each GPU kernel is shorter and the fixed CPU-side overhead becomes a larger fraction.
  2. The FP8 and AWQ INT4 paths use different kernels with different launch costs: Triton fused MoE, DeepGEMM and the activation-quantization kernels are all launched through the Python layer, which costs far more per launch than the C++ operators (Marlin / cuBLAS) AWQ uses.
- **Why do small steps lead to low GPU utilization?** GPU utilization is GPU compute time divided by total time. When launch-bound, the GPU sits idle waiting for the CPU to submit kernels. How long a kernel runs depends on how much work it contains. If there is little work per kernel, i.e. the step's batch is small and too few tokens are processed per step, each kernel finishes quickly and the GPU has to wait before the CPU has prepared the next one. Ideally the GPU is always busy and the CPU's preparation latency is hidden behind GPU compute; this is called **latency hiding**.
- **Why are steps so "small" (step here means one scheduler step of continuous batching)?** The default max-num-batched-tokens is 2048, while align mode requires prefill chunks to be multiples of 1,056, so a step holds only one prefill chunk (1,056 tokens) plus some decodes. ~1,000 tokens of compute per step is too little to cover the time the CPU needs to launch a full step of kernels; and a step that size is beyond the CUDA graph capture range (≤256), so kernels are launched eagerly one by one.

So the bottleneck moved from "the GPU can't compute fast enough" to "the CPU can't feed it fast enough". The next move follows naturally: **make each step bigger** so the fixed launch overhead is amortized.

> **What is a step?**
>
> Steps solve this problem: before modern inference engines, several requests were packed into a batch of sequences and run through autoregressive generation together. But sequences generate different numbers of tokens and some finish early. Those finished sequences stay bundled in the batch, wasting compute.
>
> ORCA called the fix **iteration-level scheduling**: the scheduler re-selects the requests to run at every iteration, advances the execution engine by one iteration only, then schedules again. Finished requests can leave immediately and newly arrived requests can join the next batch. This is the core idea of continuous batching.
>
> [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)

![Continuous batching with iteration-level scheduling in the Orca paper](/machine-learning/inference/qwen36-vllm-throughput/shared/ref_orca_continuous_batching.png)

*Source: Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf), OSDI 2022, Figure 4.*

# Step 3: raise the per-step token budget (12.0 → 15.9 req/s)

## First, find the knee on a synthetic workload

To see how many tokens per step it takes to keep the GPU fed, I first used a pure-prefill synthetic workload (every request has a different prefix, 0% hit, 4,096 + 256 input tokens, 32 output tokens, closed loop at concurrency 128) and swept max-num-batched-tokens = 2048 / 4096 / 8192 / 16384 / 32768, profiling each setting for actual tokens per step, idle time between kernels and achieved kernel TFLOPS.

![Figure 3: GPU idle time between kernels vs tokens per step. FP8 has a knee around 3,000 tokens/step](/machine-learning/inference/qwen36-vllm-throughput/en/fig03_idle_vs_tokens.png)

![Figure 4: Achieved kernel TFLOPS vs tokens per step. FP8 GEMM + MoE kernels reach ~224–230 TFLOPS at large batch](/machine-learning/inference/qwen36-vllm-throughput/en/fig04_tflops_vs_tokens.png)

- **The knee is at ~3,000 tokens per step**: FP8 GPU idle drops from 64% at 1,100 tokens/step to 6.5% at 3,290 tokens/step, and to near 0 beyond that. AWQ was never very idle and changes much less.
- **FP8 / AWQ throughput ratio** jumps from 0.83× at budget 2048 (FP8 actually slower) to 1.81× at 4096, then stays around 1.8×, exactly the theoretical expectation.
- **Checking the mechanism**: for the FP8 model, look at how long each kernel waits between being launched by the CPU and starting on the GPU. At budget 2048 the median is only 9 µs and 72% of kernels start within 50 µs: the GPU runs each kernel the moment the CPU launches it, i.e. the GPU is waiting for the CPU. At 4096 the median becomes 4.6 ms: the CPU is far ahead of the GPU, and the GPU is no longer starved.

> **🔁 To re-test**: in Figure 4, FP8's "end-to-end effective TFLOPS" (dashed) drops from ~158 to 104 TFLOPS at 32768, while kernel TFLOPS (solid) keeps rising. My guess is that at this batch size the closed-loop test makes prefill and decode batches synchronize and disrupts the request rhythm, unrelated to the kernels, but this has not been verified with an open-loop test.

## Back to the real workload: better, but not past the knee

Raising the budget from 2048 to 16384 lifted real-workload max throughput from 12.0 to 15.9 req/s. But the profiler shows only ~1,750 tokens per step and FP8 still idle 37% of the time, far from the 3,000 knee. The budget is 16384, so why isn't it used?

The reason is the shape of the real workload: each request needs ~1,080 tokens of fresh prefill on average (the part beyond the prefix hit), then ~73 decode steps. Under overload, the 128 concurrency slots are almost all held by requests that are decoding, and the scheduler can only admit a new request for prefill when one finishes generating and frees a slot (a slot is one of the requests allowed to run concurrently in a scheduler step, i.e. max_num_seqs). Roughly 1–2 new requests get in per step, so tokens per step ≈ 1.5 × 1,080 + 128 decodes ≈ 1,750, matching the measurement. **At this point tokens per step are capped by max_num_seqs, not by the token budget.**

> **Synthetic and real workloads have different bottlenecks**: in the closed-loop synthetic test above, 128 requests start prefill at the same time and tokens per step are limited only by the budget; with Poisson arrivals and short outputs, the limit becomes the number of concurrency slots. Always validate tuning conclusions on a workload with the real shape.

# Step 4: raise max_num_seqs (15.9 → 20.6 req/s)

Since concurrency slots are the new bottleneck, raise max_num_seqs from 128 to 256 and 384 (budget fixed at 16384). This model's KV cache holds ~2.1M tokens; 384 × ~3,300 tokens ≈ 1.27M fits.

![Figure 5: Raising max_num_seqs. Left: max throughput; right: GPU idle fraction, labeled with tokens per step](/machine-learning/inference/qwen36-vllm-throughput/en/fig05_max_num_seqs.png)

| max_num_seqs | FP8 max throughput | AWQ max throughput | FP8 / AWQ | FP8 tokens per step / idle | FP8 TTFT / TPOT p50 at moderate load |
|-|-|-|-|-|-|
| 128 | 15.9 req/s | 10.8 req/s | 1.48× | 1,756 / 37% | 235 / 18.0 ms |
| 256 | 19.7 req/s | 11.9 req/s | 1.66× | 3,038 / 17% | 228 / 17.0 ms |
| 384 | 20.6 req/s | 12.5 req/s | 1.65× | 10,068 / 1.7% | 288 / 23.0 ms |
| 512 | fails to start | 12.8 req/s | — | — | — |

- **FP8 gains a lot**: once tokens per step cross the 3,000 knee, idle drops from 37% to 1.7% and throughput rises 29%.
- **AWQ gains little**: AWQ already kept the GPU busy (2.3% idle); more requests only help through slightly better kernel efficiency at larger batch, +16%.
- **Diminishing returns from 256 → 384**: the GPU is barely idle by then; further gains have to come from faster kernels.

The max throughput above is a single point under overload. To plan capacity against latency targets, I also ran full arrival-rate sweeps at 128 / 256 / 384 (no profiler, 3 × 60 s per rate):

![Figure 5b: Throughput and time to first token for the three max_num_seqs settings. Top: completed throughput vs arrival rate; bottom: TTFT p99 vs completed throughput](/machine-learning/inference/qwen36-vllm-throughput/en/fig05b_seqs_rate_sweep.png)

| Config | Saturation throughput | Capacity at TTFT p99 < 1 s | Capacity at TTFT p99 < 2 s and TPOT p50 < 100 ms | TTFT / TPOT p50 at 6 req/s |
|-|-|-|-|-|
| AWQ, seqs 128 | 10.8 | 7.6 | 7.6 | 229 / 20.1 ms |
| AWQ, seqs 256 | 11.9 | 7.6 | 9.8 | 230 / 20.1 ms |
| AWQ, seqs 384 | 12.3 | 7.6 | 9.8 | 221 / 18.8 ms |
| FP8, seqs 128 | 15.0 | 13.6 | 13.6 | 259 / 20.9 ms |
| FP8, seqs 256 | 19.3 | 17.6 | 13.6 | 258 / 20.7 ms |
| **FP8, seqs 384** | **20.3** | **17.7** | **15.4** | 229 / 17.8 ms |

All units are req/s. How to read it:

- **Against a latency target, FP8 with 256/384 slots offers ~17.6 req/s of usable capacity, 2.3× AWQ.** The gap is larger under a latency constraint than at saturation (20.3 vs 12.3), because AWQ's time to first token starts climbing at around 8 req/s.
- **For AWQ, a larger max_num_seqs only absorbs peaks**: saturation rises from 10.8 to 12.3, but capacity at TTFT p99 < 1 s stays at 7.6 req/s. The GPU was already saturated, so extra admitted requests just queue.
- **Latency at moderate load is unaffected**: at 6 req/s, TTFT / TPOT are about the same for all three settings. An earlier measurement showed 35% higher TPOT at 384; it did not reproduce here and was most likely noise.
- This sweep ran on different nodes from the single-point table above; the difference in FP8 saturation at 128 (15.0 vs 15.9) comes from the node (see "Pitfalls" item 6).

> **🔁 To re-test (2 items)**
> - FP8 with max_num_seqs=512 always fails to start: torch.compile raises `KeyError: 'cubin'` while compiling the Triton kernels for GDN state indexing during profile_run; AWQ with the same config is fine. Need to find out whether it is a vLLM / Inductor issue or an environment issue, or retry on a newer version.
> - AWQ at 384 shows fewer tokens per step in the profiler window (2,936) than at 256 (6,358), contradicting the monotonic rise in throughput. Probably the 60-step window is too short and happened to land in a lull of requests entering and leaving; needs a longer window or multiple samples.

# One unifying rule: whether the GPU is fed depends only on tokens per step

Put every experiment (synthetic budget sweep, real-workload budget settings, real-workload max_num_seqs settings) on one chart, with actual tokens per step on the x-axis and GPU idle fraction on the y-axis:

![Figure 6: GPU idle fraction vs tokens per step across all configurations (log x-axis). Whichever knob was turned, the points fall on the same curve](/machine-learning/inference/qwen36-vllm-throughput/en/fig06_idle_universal.png)

Whether we changed the token budget or max_num_seqs, and whether the load was synthetic or real, the FP8 points fall on one curve: **below ~3,000 tokens per step the GPU idles because CPU launches can't keep up; above it the GPU stays busy**. So the tuning goal can be stated directly as "get steady-state tokens per step above 3,000"; which knob gets you there depends on which limit is currently binding.

# How far from the ceiling: three theoretical bounds

Throughput gains alone don't tell you how much room is left. For each step I estimated three theoretical ceilings (the three lines in Figure 1), all in req/s.

## Compute ceiling

The FLOPs per request have two parts:

- **Linear layers (MoE + full-attention projections + GDN projections)**: ~4.87 GFLOP per token (model dimensions inferred from kernel shapes). Tokens that need compute = prefill tokens that miss the cache + output tokens.

A linear layer with weight shape `[K, N]` does K×N multiply-adds per token, i.e. **2·K·N FLOP**. Add up every weight matrix a token actually passes through:

| Part | Layers | Weights per layer (K×N) | Multiply-adds per layer per token |
|-|-|-|-|
| MoE | 40 | 8 selected experts × 3 matrices (gate / up / down, each 2048×512) | 25.2 M |
| | | shared expert 3 × 2048×512 | 3.1 M |
| | | router gate 2048×256 | 0.5 M |
| | | **subtotal** | **28.8 M** |
| Full attention | 10 | q / k / v / output-gate projection 2048→9216 | 18.9 M |
| | | o_proj 4096→2048 | 8.4 M |
| | | **subtotal** | **27.3 M** |
| GDN | 30 | in_proj_qkvz 2048→12288 | 25.2 M |
| | | in_proj_ba 2048→64 | 0.13 M |
| | | out_proj 4096→2048 | 8.4 M |
| | | **subtotal** | **33.7 M** |

```
40×28.8M + 10×27.3M + 30×33.7M ≈ 1153M + 273M + 1011M ≈ 2.44 G multiply-adds
× 2 = 4.87 GFLOP / token
```

- **MoE only counts what is activated**: each token goes through its top-8 experts; the other 248 do no work. 2.44 G multiply-adds ≈ 2.4B active linear-layer parameters, which is what "A3B" means.
- **"Inferred from kernel shapes"**: the profiler trace shows N and K of every GEMM; dimensions like 9216, 12288 and 4096 were read from the trace and matched against the model config. 9216 = q 16 heads × 256 + output gate 4096 + k and v 2 heads × 256 each; 12288 = q and k 2048 each + v 4096 + z 4096.
- **Tokens that hit the prefix cache reuse their KV and GDN state and skip the linear layers entirely**, which is why prefix caching saves so many FLOPs.

- **Attention scores**: only the 10 full-attention layers count, ~`10 layers × 4 × 16 q heads × 256 dims = 163,840 ≈ 164K FLOP / (token · context position)`, multiplied by each token's context length.

For one query token and one context position, each q head does:

- **QKᵀ**: a dot product of two 256-dim vectors, 256 multiply-adds = 2×256 FLOP;
- **PV**: weight the 256-dim V by that score and accumulate, also 2×256 FLOP;
- **4×256 FLOP** in total. The multiplier is the 16 q heads, not the KV heads: GQA lets several q heads share one KV, which saves memory and bandwidth, but each q head still computes its own scores.

AWQ is counted entirely at the FP16 peak of 148 TFLOPS; for FP8, linear layers at 296 TFLOPS and attention and GDN (still BF16) at 148 TFLOPS. Compute ceiling = 1 ÷ (time needed per request).

**Sanity check**: 4.87 GFLOP/token ÷ 2 ≈ 2.44B parameters; adding the embedding and lm_head not counted here (~0.5B) gives ~2.9B, consistent with "A3B" (~3B active parameters) in the model name.

## Bandwidth ceiling

At 4 TB/s, each request's memory traffic has three parts:

- **Weights**: ~22 GiB for AWQ, ~34 GiB for FP8. They are read in full every step, so they are amortized per request by each step's prefill tokens and decode batch size. The bigger the batch, the smaller each request's share, which is why the bandwidth ceiling jumps from 67 to 157 req/s after raising max_num_seqs.
- **KV cache**: each context token takes ~20 KB across the 10 full-attention layers. Every decode step reads the whole context for each sequence, ~66 MB.
- **GDN state**: ~64 MB per sequence across 30 layers, read once and written once per decode step, ~129 MB. This traffic is specific to hybrid architectures, and it is larger than the KV traffic.

## Roofline ceiling: compute time and memory time don't simply add up

Within a kernel, the GPU computes on the current tile while asynchronously loading the next one (the latency hiding mentioned earlier), so a kernel takes roughly the longer of its compute time and memory time, not their sum. My convention is per-phase max: **for prefill and for decode separately take max(compute time, memory time), then add the two phases**. How well the two overlap depends on the implementation, so the table also lists the most optimistic (full overlap) and most pessimistic (plain sum) versions as a range:

| Bound (req/s) | S0 | S1 | S2 | S3 | S4 |
|-|-|-|-|-|-|
| Compute only (full overlap, optimistic) | 8.8 | 24.2 | 44.7 | 44.7 | 44.7 |
| Bandwidth only | 59.7 | 76.5 | 54.9 | 66.6 | 157 |
| **Per-phase roofline (main)** | **8.5** | **21.9** | **33.7** | **33.7** | **38.2** |
| Plain sum (pessimistic) | 7.7 | 18.3 | 24.6 | 26.7 | 34.7 |
| Measured | 3.9 | 9.9 | 12.0 | 15.9 | 20.6 |
| Measured / per-phase roofline | 46% | 45% | 35% | 47% | 54% |

How to read it:

- **Overall the workload is compute-bound**: at every step the bandwidth ceiling is far above the compute ceiling. The decode phase on its own is bandwidth-bound (each step moves weights, KV and GDN state but computes little), but decode is short in this workload and a small share of the total.
- **S2's share of roofline actually drops** (45% → 35%): FP8 raised the ceiling but the measurement didn't keep up; the gap is the CPU-launch holes. S3 and S4 recover it, back to 54%.
- **Where is the remaining 46%?** At S4 the GPU is barely idle, so the gap is mostly kernel efficiency: FP8 GEMMs actually run at ~215 TFLOPS (73% of peak), and attention, GDN, sampling and quantization kernels are less efficient still. The next step has to be the kernels.

# Arrival-rate sweep: throughput and latency curves

The "max throughput" numbers so far were all measured under overload. Capacity planning also needs to know whether throughput keeps up at different arrival rates and how latency behaves. Below, with max_num_seqs=128 and budget 16384, I swept from 2 to 20 req/s, 3 × 60 s per rate, with the profiler off throughout.

![Figure 7: Completed throughput vs arrival rate. Hugging the diagonal means everything that arrives gets done; flattening means saturation](/machine-learning/inference/qwen36-vllm-throughput/en/fig07_throughput_vs_rate.png)

![Figure 8: TTFT and TPOT vs completed throughput (log scale); solid = p50, dashed = p99](/machine-learning/inference/qwen36-vllm-throughput/en/fig08_latency_vs_throughput.png)

- **Saturation points**: ~10.9 req/s for AWQ and ~15.8 req/s for FP8, consistent with the separate overload measurements of 10.8 and 15.9.
- **Latency is flat before saturation and queueing explodes after**: at 10 req/s AWQ is near saturation, with TTFT p50 547 ms and TPOT p50 103 ms; FP8 is still comfortable, with TTFT p50 285 ms and TPOT p50 41 ms.
- **At low load the two are about the same**: at 2 req/s TTFT p50 is ~130–140 ms, TPOT p50 4.8 ms for FP8 and 5.3 ms for AWQ. FP8's weights are larger (~34 vs 22 GiB) but it is not slower at low load.
- **Capacity under a latency target**: with TTFT p99 < 1 s, FP8 handles ~14 req/s (p99 636 ms) and AWQ ~8 req/s (p99 769 ms): FP8 absorbs ~75% more traffic.

> **The first run of this section had wrong data**: initially I also grabbed a profile at every rate, and FP8 saturated at only 12.8 req/s with elevated latency even at low load. Digging in showed that repeatedly starting and stopping the profiler slows the FP8 engine down persistently (see "Pitfalls" item 2); the figures above are a re-run with the profiler off throughout. The AWQ sweep was also profiled at the time, but AWQ is GPU-bound, so the extra CPU-side overhead is hidden and its saturation matches the separate overload measurement; I kept it.

> **Re-tested since**: arrival-rate sweeps at max_num_seqs 256 / 384 are in Figure 5b in Step 4. The same config's saturation throughput varies 2–6% across nodes, so read the FP8 / AWQ ratios above as ±5%.

# Quality check: do prefix caching and FP8 make outputs worse?

Before shipping higher throughput we need to confirm output quality didn't drop. Prefix caching shouldn't change results in theory, but on a hybrid architecture it caches and restores GDN state snapshots, which an implementation could get wrong; FP8 genuinely changes numerical precision. So I A/B-tested both prompts in this pipeline: information summary (generates search queries from creator signals; the one load-tested in this post) and copy generation (picks a topic from the search results and writes notification copy; same model, ~4,600 input tokens, ~110 output tokens).

## How we tested

- **Samples**: for each prompt, 500 production calls from the last 7 days, stratified by language, keeping the real accept / reject ratio (~92% accept).
- **Arms**: 4 arms with exactly the same prompt version and sampling parameters, only the backend differs: base (AWQ, no prefix cache, same as production), base rerun (the same config generated again), AWQ + prefix cache, FP8 + prefix cache.
- **Why a base rerun**: both prompts sample randomly (temperature 0.6 and 1.0), so the same config produces different outputs on two runs. The gap between base rerun and base is the "random noise" reference; another arm only counts as different if it clearly exceeds that.
- **Evaluation**: LLM judges score per dimension (0/1 or a 4-point scale). For copy generation, besides format, topic selection and copy quality, I added 4 dimensions: whether the language matches the user, whether it is safe with no personal attacks, relevance to the user's features, and relevance to the search query. A separate pairwise "equivalence" judge decides whether two outputs are equivalent for downstream use. All comparisons are paired per item, reporting the mean and a 95% bootstrap confidence interval.
- **Judge rate limits**: the two prompts needed ~25,000 judge calls. Running generation and scoring together on the eval platform quickly hit the judge model's requests-per-minute limit and many scores failed. I switched to generation-only runs first (a dozen minutes or so), then scored separately with an adaptive rate-limiting script: on a rate-limit error it slows down 30% and retries, after 2 minutes without errors it speeds up. It settled at ~250 calls per minute with only 13 rate-limit errors.

![Figure 9: Paired difference vs base for each arm (mean and 95% CI). Gray is the same-config rerun, i.e. the range of random noise](/machine-learning/inference/qwen36-vllm-throughput/en/fig09_quality_forest.png)

## Findings

- **AWQ + prefix cache: no measurable change.** On both prompts every dimension, the accept rate and the equivalence rate overlap the same-config rerun.
- **FP8 + prefix cache: quality scores don't drop, but outputs change a bit more.** On information summary, FP8's equivalence to base is ~6 points lower than random noise (0.656 vs 0.716), so numerical precision does push some outputs toward other candidates, but no quality score gets worse.
- **The one business-relevant effect: FP8's accept rate on copy generation is ~4 points lower** (0.870 vs base 0.908, production 0.914); the extra rejections are mostly "no relevant candidate". These samples are borderline cases: the judge agreed with FP8's rejection in only 12% of them, and with base's original acceptance in 0%, so neither side handles them well. The single test gives p = 0.018, about 0.11 after multiple-comparison correction: leaning real but not yet conclusive.
- **Be careful with safety scores**: on information summary all three arms score ~0.05 lower on safety than base, including the same-config rerun, so base just happened to score high this time. Comparing FP8 directly to the rerun, the difference is only +0.004. Without the rerun arm it would be easy to misread this as "FP8 reduces safety".

> **🔁 To re-test**: the ~4-point lower FP8 accept rate on copy generation needs another FP8 run to see whether it reproduces. Also, absolute scores are low for every arm, including production outputs (e.g. topic selection ~0.33), suggesting the judges are strict or the prompt has a general issue. It doesn't affect the A/B conclusion but deserves its own look.

# Pitfalls

1. **Skipping warm-up gives exactly the opposite conclusion**: the first FP8 vs AWQ comparison showed FP8 46% slower than AWQ. The FP8 path's Triton kernels JIT-compile the first time they see a new shape, and that time was counted in the benchmark. After warm-up the two were even. Since then every restart gets 3 warm-up rounds.
2. **Repeatedly starting and stopping the profiler slows the engine down persistently**: after starting and stopping the torch profiler 5 times on the same server, FP8 overload throughput fell from 15.1 to 13.3 req/s (−12%) and TPOT p50 rose from 115 to 131 ms, and it never recovered; the container was not CPU-throttled and the host had idle CPUs. A single start/stop showed no effect (19.55 vs 19.69 req/s). FP8's engine scheduler thread already saturates one CPU core, so it is the most sensitive to extra CPU-side overhead. **Practice**: take throughput only from runs without the profiler; restart a server that has been profiled several times before measuring.
3. **Profiling timing and file integrity**: start capturing before load ramps up and you capture idle time; copy the trace while it is still being written and you get a corrupted gzip. I changed it to start capturing only once running + waiting requests reach a threshold, and to wait for the file size to stabilize and pass `gzip -t` before copying.
4. **Closed-loop synthetic load can mislead the bottleneck analysis**: see Step 3. With a fixed-concurrency closed loop where requests start together, the bottleneck is the token budget; with Poisson arrivals on the real workload, it becomes the concurrency slots.
5. **Not every FP8 backend is faster**: switching MoE to the DeepGEMM backend was ~30% slower (TPOT up to ~26 ms); we ended up with the default Triton backend.
6. **2–6% variance between nodes**: the same config (FP8, max_num_seqs 128) scheduled onto 4 different nodes saturated at 15.9, 15.0, 15.8 and 15.0 req/s; two measurements at 256 / 384 differed by ~2%. Read single numbers as ±5%, and run comparisons on the same node in the same time window when possible, or measure several nodes and report a range.

# Re-test list and next steps

The table below collects the places marked 🔁 in the text, plus optimizations not yet done but worth doing, ordered by impact on the conclusions.

| Item | What we saw / why it's in doubt | How to re-test or move forward |
|-|-|-|
| **FP8 accept rate on copy generation** | On 500 samples FP8 sends ~4 points less copy (p = 0.018, ~0.11 corrected); the extra rejections are "no relevant candidate" on borderline samples. | Generate another FP8 run and see whether the drop reproduces; if it does, assess the business impact. |
| **Evaluators are strict overall** | Absolute scores are low for every arm, including production outputs (e.g. topic selection ~0.33). | Sample the judges' deduction reasons to tell strict judging from a general prompt issue. |
| **FP8 fails to start at max_num_seqs=512** | torch.compile raises `KeyError: 'cubin'` compiling GDN-related Triton kernels. | Minimize the repro and report upstream; retry on newer vLLM / PyTorch. |
| **Odd AWQ tokens per step at 384** | The profiler window shows fewer tokens than at 256, contradicting the throughput trend. | Longer sampling windows, average over several samples. |
| **Effective TFLOPS drop at 32768 on the synthetic load** | End-to-end effective compute falls to ~104 TFLOPS while kernel TFLOPS rises. | Re-run that setting with an open-loop Poisson test. |
| **Intermediate budgets on the real workload** | Only 2048, 16384 and 32768 were tested; 4096 / 8192 were not. | Fill them in to find the smallest sufficient budget once max_num_seqs is raised. |
| ✅ Quality-eval sample size | Originally only 100 samples. | Done: 500 per prompt + same-config rerun, see "Quality check". |
| ✅ Node-to-node variance | Same config, different throughput on different nodes. | Done: 2–6% across 4 nodes. |
| ✅ Rate sweeps and TPOT at higher concurrency | Previously swept only at max_num_seqs=128; TPOT at moderate load was once 35% higher at 384. | Done: see Figure 5b; the TPOT increase did not reproduce. |
| Next: **tune the FP8 MoE kernel config** | vLLM ships no tuned FP8 config for this expert shape (E=256, N=512) on H20, so it uses Triton defaults. | Generate a config with vLLM's `benchmark_moe.py --tune` and load it; should directly improve MoE kernel efficiency. |
| Next: **CUDA graphs for prefill steps** | Only steps of ≤256 tokens use CUDA graphs; every prefill step launches eagerly. | Evaluate the gain and memory cost of a larger capture range or piecewise graphs. |
| Next: **align the prefix to 1,056** | The shared prefix hits only 2,112 / 2,570 tokens. | Adjust the system prompt length to a block boundary; the hit rate should rise above 66%. |

# Summary

- **First look for compute you're doing for nothing**: a long shared prefix + prefix caching is the cheapest 2.5×. On hybrid architectures, watch for the whole-block hit restriction from the block size and align mode.
- **Whether quantization pays off depends on the bottleneck**: FP8 made kernels 1.7× faster, but when each step carries too little compute the bottleneck moves to CPU launches and eats the end-to-end gain.
- **Watch one metric, tokens per step**: it decides whether the GPU is fed. The token budget and max_num_seqs are just levers; tune whichever one is limiting it, and judge that on a workload with the real shape.
- **Use the roofline to know what's left**: we ended at 54% of the ceiling with the GPU no longer idle; the remaining room is in kernel efficiency.
- **Quality checks need a noise reference**: with sampled outputs, the same config differs between two runs. A same-config rerun as the reference plus per-item paired comparison is what separates real regressions from noise. Here it ruled out a seemingly significant "safety drop", and surfaced a real open question: FP8 sends ~4% less copy on copy generation.
