---
title: "NVIDIA Vera Rubin + LPX: Why Move FFN to LPU?"
description: Deep dive into the design rationale behind NVIDIA Vera Rubin's GPU + LPX collaborative inference, from Roofline Model to Memory-Bound analysis
tags: [nvidia, inference, hardware, roofline-model, memory-bound, vera-rubin]
---

# NVIDIA Vera Rubin + LPX: GPU and LPU Collaborative Inference

> In 2025, NVIDIA released the Vera Rubin architecture, featuring a key innovation: the **LPX (Linear Processor Accelerator)**. During inference, **Attention is computed on GPU, while FFN is computed on LPU**. Why split it this way? This involves core concepts of **Roofline Model**, **Arithmetic Intensity**, and **Memory-Bound vs Compute-Bound**.

## Two Phases of Transformer Inference

LLM inference consists of two distinct phases:

| | Prefill | Decode (Token-by-Token Generation) |
|---|---|---|
| **Input** | Entire prompt (e.g., 1024 tokens) | 1 token from previous step |
| **Compute Pattern** | Large batch matrix multiplication | Token-by-token matrix-vector multiplication |
| **Bottleneck** | Compute (Compute-Bound) | Memory bandwidth (Memory-Bound) |
| **GPU Utilization** | High (large matrix parallelism) | Extremely low (massive compute idle waiting for data) |

<HtmlVisualization
  src="/machine-learning/inference/visualizations/prefill-vs-decode-en.html"
  height="520px"
  title="Prefill vs Decode: Two Distinct Phases"
/>

::: info Key Difference
The **Prefill** phase processes the entire prompt, packing all tokens into a large matrix for batch matrix multiplication—GPU's thousands of cores can run at full capacity.

In the **Decode** phase, only one token is generated at a time. The weight matrix $W$ size remains the same (e.g., $4096 \times 16384$), but the input degrades from a matrix to a vector ($1 \times 4096$), wasting massive compute power.
:::

## What is Arithmetic Intensity?

**Arithmetic Intensity** is the core metric for understanding GPU utilization:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs (Compute)}}{\text{Bytes (Data Movement)}}
$$

It answers a simple question: **How many computations can be performed per byte of data moved?**

### FFN Arithmetic Intensity in Decode Phase

Take a typical FFN layer as example (hidden_dim = 4096, intermediate_dim = 16384, FP16):

**Weight size:**
- $W_1$: $4096 \times 16384 \times 2 \text{ bytes} = 128 \text{ MB}$
- $W_2$: $16384 \times 4096 \times 2 \text{ bytes} = 128 \text{ MB}$
- Total loading from VRAM: **~256 MB**

**Compute (Batch Size = 1):**
- $W_1$: $4096 \times 16384 \times 2 = 134M \text{ FLOPs}$
- $W_2$: $16384 \times 4096 \times 2 = 134M \text{ FLOPs}$
- Total: **~268M FLOPs**

$$
\text{AI} = \frac{268 \times 10^6 \text{ FLOPs}}{256 \times 10^6 \text{ Bytes}} \approx 1.05 \text{ FLOPs/Byte}
$$

::: warning What Does This Mean?
Arithmetic Intensity ≈ 1 means for every byte of data moved from VRAM, only 1 floating-point operation is performed. GPU's compute power is completely unused—**data movement time far exceeds computation time**.

Compare with Prefill phase: If batch has 1024 tokens, the same 256 MB of weights can perform $268M \times 1024 ≈ 274G$ computations, AI reaches ~1075, fully utilizing GPU's compute power.
:::

## Roofline Model: Understanding Bottlenecks at a Glance

**Roofline Model** is a classic tool for analyzing hardware performance bottlenecks. It tells us: given an operation's Arithmetic Intensity, what limits the actual performance?

<HtmlVisualization
  src="/machine-learning/inference/visualizations/roofline-model-en.html"
  height="560px"
  title="Roofline Model: Memory-Bound vs Compute-Bound"
/>

### How to Determine if an Operation is Memory-Bound or Compute-Bound?

Calculate the **Ridge Point**:

$$
\text{Ridge Point} = \frac{\text{Peak Compute (FLOPs/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}}
$$

Taking **NVIDIA H100 SXM** as example:
- Peak FP16 Compute: **989 TFLOPS**
- HBM3 Bandwidth: **3.35 TB/s**
- Ridge Point = $989 / 3.35 ≈ 295 \text{ FLOPs/Byte}$

| Operation | Arithmetic Intensity | vs Ridge Point | Bottleneck Type |
|---|---|---|---|
| FFN Decode (BS=1) | ~1 FLOPs/Byte | 1 ≪ 295 | **Memory-Bound** |
| FFN Prefill (BS=1024) | ~1075 FLOPs/Byte | 1075 > 295 | **Compute-Bound** |
| Attention (Decode) | Variable, usually low | Usually < 295 | **Memory-Bound** |

## GPU + LPX Collaborative Architecture

Understanding the Memory-Bound problem reveals NVIDIA's design intent:

<HtmlVisualization
  src="/machine-learning/inference/visualizations/gpu-lpx-architecture-en.html"
  height="620px"
  title="Vera Rubin GPU + LPX Collaborative Inference Architecture"
/>

### Why is FFN Better on LPU?

LPU (Linear Processing Unit) is designed specifically for **Memory-Bound linear operations**:

| Feature | GPU | LPU (LPX) |
|---|---|---|
| **Design Goal** | Large-scale parallel compute | High-bandwidth linear operations |
| **Compute Power** | Extremely strong (~1000 TFLOPS) | Moderate |
| **Memory Bandwidth** | High but insufficient (~3 TB/s) | Extremely high (bandwidth-optimized) |
| **Suitable For** | Compute-Bound operations | Memory-Bound operations |
| **Utilization at AI = 1** | < 1% (massive compute idle) | High (compute matches bandwidth) |

::: info Core Insight
GPU's compute/bandwidth ratio (Ridge Point) is too high—for operations with AI ≈ 1, GPU wastes 99% of compute power waiting for data.

LPU's design philosophy is to **lower the Ridge Point**: not pile on compute, but pile on bandwidth, making the compute-to-bandwidth ratio match FFN Decode's actual needs.

This is like using a sports car (GPU) vs a truck (LPU) for delivery—the sports car is faster but can only carry one item at a time, the truck is slower but has high throughput. When the bottleneck is "moving things" not "speed," the truck is more suitable.
:::

### Data Movement Overhead

After GPU finishes Attention, it does need to move intermediate results to LPU for FFN, then move back. This adds extra latency, but:

1. **Small data volume**: Intermediate activations are just one vector (e.g., $1 \times 4096 \times 2 = 8 \text{ KB}$), while FFN weights are 256 MB
2. **High-speed interconnect**: Vera Rubin architecture uses NVLink and other high-speed buses to connect GPU and LPX
3. **Pipeline parallelism**: FFN computation of layer $n$ can overlap with Attention computation of layer $n+1$

$$
\text{Movement overhead} = \frac{8 \text{ KB}}{\text{hundreds of GB/s bandwidth}} \approx \text{tens of nanoseconds}
$$

Compared to millisecond-level latency from FFN being Memory-Bound on GPU, movement overhead is negligible.

## Complete Process for Determining Memory-Bound vs Compute-Bound

```
1. Calculate FLOPs for the operation (total floating-point operations)
2. Calculate Bytes to move (weights + input + output)
3. Arithmetic Intensity = FLOPs / Bytes
4. Check hardware's Ridge Point = Peak Compute / Peak Bandwidth
5. If AI < Ridge Point → Memory-Bound (bottleneck is data movement)
   If AI > Ridge Point → Compute-Bound (bottleneck is computation)
```

::: warning Considerations in Real Engineering
- **Larger Batch Size = Higher AI**: Increasing Batch Size from 1 to N, weights are read once but computed N times, AI grows linearly
- **Quantization reduces weight size**: FP16 → INT8 → INT4 reduces data movement, improving AI
- **KV Cache** is an additional Memory-Bound factor in Attention phase
- In real systems, Prefill and Decode may be mixed (e.g., continuous batching)
:::

## Summary

| Question | Answer |
|---|---|
| Why put FFN on LPU? | FFN in Decode phase is Memory-Bound (AI ≈ 1), GPU wastes 99% compute, LPU's bandwidth/compute ratio is better matched |
| Isn't FFN also matrix computation? | Yes, but in Decode it degrades to matrix×vector, small compute volume but large data movement |
| Why is AI so low? | Only 1 token processed at a time, 256 MB weights perform only 268M computations |
| Doesn't movement take time? | Moving 8 KB activations, not 256 MB weights—tens of nanoseconds with high-speed interconnect |
| How to determine Memory/Compute Bound? | Calculate AI, compare with Ridge Point. AI < Ridge Point = Memory-Bound |
