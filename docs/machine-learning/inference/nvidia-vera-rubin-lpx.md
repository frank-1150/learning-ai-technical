---
title: "NVIDIA Vera Rubin + LPX: 为什么 FFN 要搬到 LPU 上？"
description: 深入理解 NVIDIA Vera Rubin 架构中 GPU + LPX 协同推理的设计原理，从 Roofline Model 到 Memory-Bound 分析
tags: [nvidia, inference, hardware, roofline-model, memory-bound, vera-rubin]
---

# NVIDIA Vera Rubin + LPX：GPU 与 LPU 协同推理

> 2025 年 NVIDIA 发布了 Vera Rubin 架构，其中一个关键创新是引入了 **LPX（Linear Processor Accelerator）**。推理时 **Attention 在 GPU 上计算，FFN 在 LPU 上计算**。为什么要这样拆分？这背后涉及到 **Roofline Model**、**Arithmetic Intensity** 和 **Memory-Bound vs Compute-Bound** 的核心概念。

## Transformer 推理的两个阶段

LLM 的推理分为两个截然不同的阶段：

| | Prefill（预填充） | Decode（逐 Token 生成） |
|---|---|---|
| **输入** | 整个 prompt（如 1024 tokens） | 上一步生成的 1 个 token |
| **计算方式** | 大批量矩阵乘法 | 逐 token 的矩阵-向量乘法 |
| **瓶颈** | 计算量（Compute-Bound） | 内存带宽（Memory-Bound） |
| **GPU 利用率** | 高（大矩阵并行） | 极低（大量算力空转等数据） |

<HtmlVisualization
  src="/machine-learning/inference/visualizations/prefill-vs-decode.html"
  height="520px"
  title="Prefill vs Decode：两个截然不同的阶段"
/>

::: info 关键区别
**Prefill** 阶段处理整个 prompt，可以把所有 token 打包成一个大矩阵做 batch 矩阵乘法，GPU 的数千个核心可以满载运行。

**Decode** 阶段每次只生成一个 token，权重矩阵 $W$ 的大小不变（比如 $4096 \times 16384$），但输入从矩阵退化为一个向量（$1 \times 4096$），大量算力被浪费。
:::

## 什么是 Arithmetic Intensity？

**Arithmetic Intensity（算术强度）** 是理解 GPU 利用率的核心指标：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs（计算量）}}{\text{Bytes（数据搬运量）}}
$$

它回答一个简单问题：**每搬运 1 字节数据，能做多少次计算？**

### FFN 在 Decode 阶段的算术强度

以一个典型的 FFN 层为例（hidden_dim = 4096, intermediate_dim = 16384, FP16）：

**权重大小：**
- $W_1$: $4096 \times 16384 \times 2 \text{ bytes} = 128 \text{ MB}$
- $W_2$: $16384 \times 4096 \times 2 \text{ bytes} = 128 \text{ MB}$
- 总计需从显存加载：**~256 MB**

**计算量（Batch Size = 1）：**
- $W_1$: $4096 \times 16384 \times 2 = 134M \text{ FLOPs}$
- $W_2$: $16384 \times 4096 \times 2 = 134M \text{ FLOPs}$
- 总计：**~268M FLOPs**

$$
\text{AI} = \frac{268 \times 10^6 \text{ FLOPs}}{256 \times 10^6 \text{ Bytes}} \approx 1.05 \text{ FLOPs/Byte}
$$

::: warning 这意味着什么？
Arithmetic Intensity ≈ 1 意味着每从显存搬 1 字节数据，只做 1 次浮点运算。GPU 的算力完全用不上——**搬数据的时间远远大于计算的时间**。

对比 Prefill 阶段：如果 batch 有 1024 个 token，同样的 256 MB 权重可以做 $268M \times 1024 ≈ 274G$ 次计算，AI 达到 ~1075，GPU 的算力可以充分利用。
:::

## Roofline Model：一张图看懂瓶颈

**Roofline Model** 是分析硬件性能瓶颈的经典工具。它告诉我们：给定一个操作的 Arithmetic Intensity，实际性能受什么限制。

<HtmlVisualization
  src="/machine-learning/inference/visualizations/roofline-model.html"
  height="560px"
  title="Roofline Model：Memory-Bound vs Compute-Bound"
/>

### 如何判断一步操作是 Memory-Bound 还是 Compute-Bound？

计算**临界点（Ridge Point）**：

$$
\text{Ridge Point} = \frac{\text{Peak Compute (FLOPs/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}}
$$

以 **NVIDIA H100 SXM** 为例：
- Peak FP16 Compute: **989 TFLOPS**
- HBM3 Bandwidth: **3.35 TB/s**
- Ridge Point = $989 / 3.35 ≈ 295 \text{ FLOPs/Byte}$

| 操作 | Arithmetic Intensity | 与 Ridge Point 比较 | 瓶颈类型 |
|---|---|---|---|
| FFN Decode (BS=1) | ~1 FLOPs/Byte | 1 ≪ 295 | **Memory-Bound** |
| FFN Prefill (BS=1024) | ~1075 FLOPs/Byte | 1075 > 295 | **Compute-Bound** |
| Attention (Decode) | 变化大，通常较低 | 通常 < 295 | **Memory-Bound** |

## GPU + LPX 协同架构

理解了 Memory-Bound 问题，就能理解 NVIDIA 的设计意图：

<HtmlVisualization
  src="/machine-learning/inference/visualizations/gpu-lpx-architecture.html"
  height="620px"
  title="Vera Rubin GPU + LPX 协同推理架构"
/>

### 为什么 FFN 放在 LPU 上更好？

LPU（Linear Processing Unit）是专门为 **Memory-Bound 线性运算** 设计的：

| 特性 | GPU | LPU (LPX) |
|---|---|---|
| **设计目标** | 大规模并行计算 | 高带宽线性运算 |
| **算力** | 极强（~1000 TFLOPS） | 适中 |
| **内存带宽** | 高但不够用（~3 TB/s） | 极高（专为带宽优化） |
| **适合场景** | Compute-Bound 操作 | Memory-Bound 操作 |
| **AI = 1 时利用率** | < 1%（大量算力空转） | 高（算力匹配带宽） |

::: info 核心洞察
GPU 的算力/带宽比值（Ridge Point）太高——对于 AI ≈ 1 的操作，GPU 99% 的算力在空转等数据。

LPU 的设计思路是**降低 Ridge Point**：不堆算力，而是堆带宽，让算力和带宽的比值更匹配 FFN Decode 的实际需求。

这就像用跑车（GPU）送快递 vs 用货车（LPU）送快递——跑车更快但一次只能装一件，货车虽慢但吞吐量大。当瓶颈是"搬东西"而不是"速度"时，货车更合适。
:::

### 数据搬运的开销

每次 GPU 算完 Attention，确实需要把中间结果搬到 LPU 做 FFN，然后再搬回来。这引入了额外延迟，但：

1. **搬运量小**：中间激活值只有一个向量（如 $1 \times 4096 \times 2 = 8 \text{ KB}$），而 FFN 权重是 256 MB
2. **高速互联**：Vera Rubin 架构使用 NVLink 等高速总线连接 GPU 和 LPX
3. **流水线并行**：第 $n$ 层的 FFN 计算可以和第 $n+1$ 层的 Attention 计算重叠

$$
\text{搬运 overhead} = \frac{8 \text{ KB}}{数百 \text{ GB/s 带宽}} \approx \text{几十纳秒}
$$

相比 FFN 在 GPU 上 Memory-Bound 导致的毫秒级延迟，搬运开销可以忽略不计。

## 判断 Memory-Bound vs Compute-Bound 的完整流程

```
1. 计算该操作的 FLOPs（总浮点运算数）
2. 计算该操作需要搬运的 Bytes（权重 + 输入 + 输出）
3. Arithmetic Intensity = FLOPs / Bytes
4. 查看硬件的 Ridge Point = Peak Compute / Peak Bandwidth
5. 如果 AI < Ridge Point → Memory-Bound（瓶颈在搬数据）
   如果 AI > Ridge Point → Compute-Bound（瓶颈在计算）
```

::: warning 实际工程中的考量
- **Batch Size 越大，AI 越高**：Batch Size 从 1 增到 N，权重只读一次但计算 N 倍，AI 线性增长
- **量化降低权重大小**：FP16 → INT8 → INT4 可以减少搬运量，提高 AI
- **KV Cache** 是 Attention 阶段的额外 Memory-Bound 因素
- 实际系统中 Prefill 和 Decode 可能混合调度（如 continuous batching）
:::

## 总结

| 问题 | 答案 |
|---|---|
| 为什么 FFN 放 LPU？ | Decode 阶段 FFN 是 Memory-Bound（AI ≈ 1），GPU 算力浪费 99%，LPU 的带宽/算力比更匹配 |
| FFN 不也是矩阵计算吗？ | 是，但 Decode 时退化为矩阵×向量，计算量小而数据搬运量大 |
| AI 为什么这么低？ | 每次只处理 1 个 token，256 MB 权重只做 268M 次计算 |
| 搬运不花时间吗？ | 搬的是 8 KB 激活值，不是 256 MB 权重，高速互联下几十纳秒 |
| 怎么判断 Memory/Compute Bound？ | 算 AI，比 Ridge Point。AI < Ridge Point 就是 Memory-Bound |
