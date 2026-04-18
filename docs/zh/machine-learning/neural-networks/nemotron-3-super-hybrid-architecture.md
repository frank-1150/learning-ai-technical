---
date: 2026-04-16
title: "Nemotron 3 Super：Mamba + Transformer + Latent MoE 的混合架构"
description: 2026 年 3 月 NVIDIA 发布的 Nemotron 3 Super 是第一个在 1M context + agentic reasoning 上同时跑出 SOTA 的开源混合架构。本文拆解它的三种 building block 以及 Latent MoE 的设计哲学
tags: [llm, architecture, mamba, moe, state-space-model, nemotron, hybrid-architecture]
---

# Nemotron 3 Super：Mamba + Transformer + Latent MoE 的混合架构

> 2026 年 3 月，NVIDIA 开源了 **Nemotron 3 Super**——一个 120B 总参数 / 12B 激活参数的混合架构模型，原生支持 **1M token context**，在 agentic reasoning 基准 PinchBench 上拿到 **85.6%**，成为同级别开源模型中的最高分。本文拆解它背后的三种 building block，以及为什么 2026 年的主流架构不再是纯 Transformer。

## 2026 年的架构变局

过去 7 年 LLM 基本都建在 Transformer 之上。但 2026 年 Q1 有一批明显的信号，表明 **主流架构正在离开纯 Transformer**：

| 模型 | 发布时间 | 架构类型 | 总参数 |
|---|---|---|---|
| Jamba 1.5（AI21） | 2024 | Mamba + Attention + MoE | 398B |
| Hunyuan-TurboS（腾讯） | 2026-01 | Transformer + Mamba-2 + MoE | 560B |
| **Nemotron 3 Super（NVIDIA）** | **2026-03** | **Mamba-2 + Attention + Latent MoE** | **120B** |
| Falcon H3（TII） | 2026-02 | Attention-free（纯 SSM） | 40B |

核心驱动力只有一个：**长 context 下 Transformer 的二次复杂度太贵了**。

对于 1M token context，标准注意力要算 $O(N^2) = 10^{12}$ 次操作，存储 $O(N^2)$ 的 attention matrix。即便是最优化的 FlashAttention，在推理延迟和显存占用上都撑不住。

**Mamba 类 State Space Models（SSM）** 的优势是 **$O(N)$ 复杂度**——序列再长，每 token 的计算/存储都是常数。但 SSM 在 **精确检索（needle-in-haystack）** 类任务上做不过注意力。

**混合架构** 的思路：两种各取所长。

## 基础 1：Mamba-2 在做什么

**State Space Model（SSM）** 的直觉可以类比 **RNN + 可学习的动态系统**。标准 SSM 把序列处理看作线性动态系统：

$$
h_t = A h_{t-1} + B x_t
$$
$$
y_t = C h_t
$$

其中 $h_t$ 是隐藏状态，$x_t$ 是当前 token，$A, B, C$ 是可学习矩阵。

**Mamba 的关键创新**是 **selective SSM**：让 $B, C$ 不是固定参数，而是 **依赖于当前 token 的函数**：

$$
B_t = f_B(x_t), \quad C_t = f_C(x_t)
$$

这让模型可以根据当前 token **选择性地关注或忽略** 过去的历史——这是 Mamba 第一次让 SSM 在语言任务上跑得过 Transformer 的关键。

**Mamba-2** 在 Mamba 的基础上引入了 **结构化对角矩阵 $A$** 和 **矩阵化并行训练**——既保留了推理时的 $O(N)$ 复杂度，又让训练可以用矩阵乘法加速（而不是逐 token 扫描）。

### Mamba-2 vs Attention 的具体对比

| 特性 | Attention | Mamba-2 |
|---|---|---|
| 推理复杂度 | $O(N^2)$ 或 $O(N)$（with KV cache） | $O(N)$ |
| 训练复杂度 | $O(N^2)$ | $O(N \log N)$ |
| 长序列显存 | $O(N)$（KV cache） | $O(1)$（固定隐状态） |
| 精确检索能力 | 强 | 弱 |
| 需要 position encoding | 是 | 否（隐含在状态演化中） |

## 基础 2：Latent MoE 的压缩路由

Nemotron 3 Super 的另一个大创新是 **Latent MoE**——把 Mixture-of-Experts 做到了更极致的参数效率。

### 传统 MoE 的问题

标准 MoE（如 Mixtral、DeepSeek-V3）的工作方式：

1. 一个 gating network 决定每个 token 路由到哪几个 expert
2. 被选中的 expert 执行完整维度的计算（通常 $d_{model} = 4096$）
3. 结果加权求和

**问题**：expert 必须在完整 $d_{model}$ 维度上运行，所以 expert 越多，参数量 $\propto N_{experts} \times d_{model}^2$ 增长极快。

### Latent MoE 的做法

Nemotron 3 Super 在 expert 路由前加了一个 **降维压缩**：

```
token (4096-dim) 
    ↓ project down
  latent (1024-dim)
    ↓ route to experts
  experts (work in 1024-dim)
    ↓ combine
  latent result (1024-dim)
    ↓ project up
  output (4096-dim)
```

在压缩后的低维 latent 空间里做 expert 计算，然后投影回原维度。

![标准 MoE vs Latent MoE：在更低维度跑 expert，同计算预算下容纳 4x 专家数（来源：NVIDIA Developer Blog）](./nemotron-figures/latent-moe.png)

**关键优势**：

- 相同计算预算下，可以多出 **4x 的 expert 数量**
- 每个 expert 更小，更容易学到专一化的特征
- 投影矩阵是少量共享参数，不破坏参数效率

## Nemotron 3 Super 的层叠结构

整个模型由 **5 组重复 block** 堆叠而成。每个 block 固定是 6 层：

```
Block 结构（每 block 6 层，共 5 次重复）：
  Layer 1: Mamba-2
  Layer 2: Latent MoE
  Layer 3: Mamba-2
  Layer 4: Attention       ← 唯一的 attention 层
  Layer 5: Mamba-2
  Layer 6: Latent MoE
```

![Nemotron 3 Super 层叠结构：5 组 block × 每 block 6 层（来源：NVIDIA Developer Blog）](./nemotron-figures/layer-architecture.png)

**设计哲学**：

- **Mamba-2 层数最多（每 block 3 层）**：承担大部分序列处理，保持 $O(N)$ 推理复杂度
- **Attention 层稀疏插入（每 block 1 层，共 5 层）**：负责精确 recall 和长距离依赖
- **Latent MoE 层（每 block 2 层）**：提供参数规模和专一化能力

### 为什么 Attention 只放在中间位置？

这是一个经验工程决定。研究团队发现：

- 放在太靠前的位置：模型还没建立足够的 contextual representation，attention 不够准确
- 放在太靠后：attention 的结果被后续 Mamba 层过度压缩
- **中间位置（layer 4）是最优平衡点**：Mamba 已经提炼出好的表示，attention 可以做精确的关联，后续 Mamba 还有足够的能力聚合

这和 **Anthropic、OpenAI 的 interpretability 研究** 结论一致：中层是 LLM 形成「意义」的关键位置。

## Multi-Token Prediction（MTP）：解码加速

Nemotron 3 Super 还用了 **MTP**——每次前向传播预测未来多个 token，而不是只预测下一个。

**传统自回归解码**：

```
x₁ x₂ x₃ → predict x₄  (1 forward)
x₁ x₂ x₃ x₄ → predict x₅  (1 forward)
```

**MTP 解码**：

```
x₁ x₂ x₃ → predict x₄, x₅, x₆  (1 forward)
```

**设计细节**：

- **共享权重的 MTP 头**：所有预测头共用底层参数，只有最后一层 projection 不同
- **训练时联合优化**：4 个预测头同时训练，相当于多任务学习
- **推理时 speculative verification**：用生成的 x₄, x₅, x₆ 做 speculative decoding，只有都被接受才跳过 forward

**实测效果**：在结构化生成任务（代码、工具调用）上 **3x wall-clock 加速**。

## 训练配方

| 阶段 | 规模 | 说明 |
|---|---|---|
| 预训练 | **25T tokens（NVFP4 精度）** | 其中 10T 是独特 curated 数据 + 10B 专门 reasoning + 15M 编程题 |
| SFT | 7M 样本 | 从 40M 候选中筛选，覆盖 reasoning / 指令 / 代码 / 安全 / multi-step agent |
| RL | 21 个环境配置 | 通过 NeMo Gym + NeMo RL 生成 1.2M rollouts |

**值得注意的细节**：

- 预训练直接用 **NVFP4** 精度——这在 2026 年之前是不可行的，因为没有足够稳定的训练框架。Rubin 硬件和第三代 Transformer Engine 让这成为可能
- RL 环境多样化，包括工具使用、代码执行、多轮对话——不是单纯的 RLHF

## 基准表现

| 任务 | Nemotron 3 Super | GPT OSS 120B | Qwen3 122B |
|---|---|---|---|
| PinchBench（agentic） | **85.6%** | 78.2% | 74.5% |
| 推理吞吐（相比前代 Nemotron Super） | **5x** | — | — |

在同级别开源模型里，Nemotron 3 Super 在 agentic reasoning 上是 **最强的**——这正是它的设计目标：**为长链路 agent 任务优化**。

![Nemotron 3 Super 与 GPT OSS 120B / Qwen3 122B 的对比基准（来源：NVIDIA Developer Blog）](./nemotron-figures/benchmarks.png)

## 为什么这是"下一代"架构

Nemotron 3 Super 的意义在于它同时解决了 2026 年 LLM 发展的 **三个核心痛点**：

### 痛点 1：Thinking Tax

推理时 scaling（o1、DeepSeek-R1 那种）带来了新的成本问题——模型要「想」得更久，每个用户请求消耗的算力大幅增加。

**Nemotron 3 的回答**：混合架构让推理吞吐比前代提升 5x。"想" 的成本下降，就能承受更多 reasoning 步骤。

### 痛点 2：Context Explosion

Agent 任务需要维护长历史上下文——工具返回、中间推理、规划状态……轻易超过 100K。

**Nemotron 3 的回答**：原生 1M context + Mamba-2 的 $O(N)$ 复杂度，让长 context 不再是 premium 功能。

### 痛点 3：Expert Specialization Trade-off

传统 MoE 专家越多参数越爆炸，少了又学不到足够专一化的能力。

**Nemotron 3 的回答**：Latent MoE 在压缩空间里跑 expert，让 "多专家 + 低成本" 第一次成立。

## 我的看法

如果我用一句话概括 2026 年 LLM 架构的变化，那就是：

> **Transformer 不再是唯一答案，混合架构成为主流。**

但这不是 Transformer 的替代，而是 Transformer + SSM + 稀疏计算的 **融合**：

- Mamba-2 解决长 context 的复杂度问题
- Attention 解决精确检索问题
- Latent MoE 解决参数效率问题
- MTP 解决解码吞吐问题

下一步值得关注的方向：

1. **Mamba-3**：2026-04 在 OpenReview 出现的改进版本，进一步提升表达力
2. **Hybrid 的最优配比**：Mamba 和 Attention 按什么比例混最好？学界目前没有共识
3. **训练后端的成熟**：NVFP4 原生训练、RL 环境并行化，这些基础设施现在只有少数团队掌握

如果你对 LLM 架构演化感兴趣，推荐顺序阅读：

1. [3Blue1Brown: What is ChatGPT doing](./chatgpt-overview-3blue1brown.md) —— Transformer 的基础
2. 本文 —— 混合架构 2026 的现状
3. [vLLM & PagedAttention](../inference/vllm-pagedattention.md) —— 这些架构如何被推理引擎支持

## 参考资料

- [Introducing Nemotron 3 Super（NVIDIA Developer Blog）](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-2: Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Jamba-1.5 Hybrid Transformer-Mamba（AI21）](https://www.ai21.com/blog/announcing-jamba/)
- [Hybrid Architectures for Language Models: Systematic Analysis and Design Insights](https://arxiv.org/html/2510.04800v1)
- [Attention was never enough: Tracing the rise of hybrid LLMs（AI21）](https://www.ai21.com/blog/rise-of-hybrid-llms/)
