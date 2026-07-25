---
date: 2026-07-24
title: "训练一个 GPT-2 XL 到底要多少显存和算力？——用 AdamW 把账算到底"
description: 以 CS336 Assignment 1 的 GPT-2 XL 为例，逐项拆解训练时的显存占用（参数/梯度/优化器状态/激活值）与 FLOPs 开销，推导出 16 bytes/param 和 6ND 两个万能估算公式，并回答"单卡放不下 batch=1024 怎么办"
tags: [cs336, adamw, memory, flops, gpt-2, training, gradient-accumulation, data-parallel]
---

# 训练一个 GPT-2 XL 到底要多少显存和算力？

> 本文是 CS336 Assignment 1 第四章 *Training LLM* 中一道题的学习笔记。题目本身很朴素：给定 GPT-2 XL 的超参数，写出训练时显存占用和 FLOPs 关于 batch size 的表达式（在这之前还有手写实现 AdamW，便于分析训练占用显存多少）。
>
> 做完这个简单的计算，我觉得就能够了解训练一个模型的时候单卡在跑什么？为什么单卡要这么久？为什么必须要多卡？为了实现多卡需要什么技巧？相当于一次 nGPT 训练所需要的resource accounting的计算了

## 模型配置

我们用的是 GPT-2 XL 规模的模型（CS336 的实现里用了 RMSNorm + SwiGLU，和原版 GPT-2 的 LayerNorm + GELU 略有不同）：

| 符号 | 含义 | 取值 |
|---|---|---|
| $V$ | 词表大小 | 50,257 |
| $T$ | 上下文长度 | 1,024 |
| $L$ | Transformer 层数 | 48 |
| $D$ | 模型维度 `d_model` | 1,600 |
| $H$ | 注意力头数 | 25 |
| $d_{ff}$ | FFN 中间维度 | 4,288 |

::: tip 为什么 d_ff = 4288 而不是 4×1600？
SwiGLU 有**三个**权重矩阵（$W_1, W_2, W_3$）而不是普通 FFN 的两个。为了让参数量和 4× 的普通 FFN 持平，业界惯例是取 $d_{ff} \approx \frac{8}{3} D$，再向上取整到 64 的倍数（对齐 GPU tensor core）：$\frac{8}{3} \times 1600 = 4266.7 \rightarrow 4288$。
:::

## 一、参数量：约 16.4 亿

先数参数。每个 Transformer block：

$$
\underbrace{4D^2}_{Q,K,V,O} + \underbrace{3 \cdot d_{ff} \cdot D}_{\text{SwiGLU}} + \underbrace{2D}_{\text{两个 RMSNorm}} = 30{,}825{,}600
$$

| 组成部分 | 参数量 |
|---|---|
| 单个 block | 30,825,600 |
| × 48 层 | 1,479,628,800 |
| Embedding + LM Head + 最终 RMSNorm | 160,824,000 |
| **合计** | **1,640,452,800** |

注意最后一行：embedding 和 lm_head 各 $VD = 8000$ 万参数，两者加起来 1.6 亿，**占了总参数的 10%**，但它们只参与一次矩阵乘。这也是为什么很多模型选择 **weight tying**（共享 embedding 和 lm_head 权重）——一下省掉 5% 的参数、梯度和优化器状态。

## 二、显存：16 bytes/param 这个魔法数字

假设全程 float32（4 bytes）。训练时显存分四块：

| 类别 | 表达式 | 数值 |
|---|---|---|
| 参数 | $4 \times P$ | 6.56 GB |
| 梯度 | $4 \times P$ | 6.56 GB |
| 优化器状态（$m$ 和 $v$） | $8 \times P$ | 13.12 GB |
| **小计（与 batch size 无关）** | $\mathbf{16 \times P}$ | **26.25 GB** |
| 激活值 | $4L(7BTD + BHT^2 + 3BT \cdot d_{ff})$ | **9.76·B GB** |

::: important 记住这个数：fp32 + AdamW = 16 bytes/param
参数 4 + 梯度 4 + AdamW 的一阶动量 $m$ 4 + 二阶动量 $v$ 4 = **16 bytes**。

也就是说，**模型本身还没开始算，显存就已经是参数量的 16 倍**。1.64B 的模型光静态开销就 26 GB。这就是为什么 7B 模型你根本没法在 24GB 的 4090 上做全参数微调（7×16 = 112 GB），而推理只要 14 GB（fp16）。

如果换成 SGD（无动量），这个数字是 8 bytes/param；带 momentum 的 SGD 是 12。AdamW 的收敛优势是用 2× 参数量的额外显存换来的。
:::

### 激活值那一项是怎么来的

每层需要为 backward 保存的中间结果（单位：float 个数）：

- $7BTD$ —— RMSNorm 输入、$Q$/$K$/$V$、注意力输出、残差、FFN 相关的若干个 $B \times T \times D$ 张量
- $BHT^2$ —— 注意力分数矩阵 $QK^\top$，形状 $B \times H \times T \times T$
- $3BT \cdot d_{ff}$ —— SwiGLU 三个分支的中间激活

代入 $B=1$：每层 50,855,936 个 float，48 层共 24.4 亿个，乘 4 bytes = **9.76 GB**。

::: warning 这是一个粗略下界
不同实现保存的中间张量数量差别很大（有没有 fused kernel、softmax 输出存不存、dropout mask 等），$7BTD$ 这个系数是量级估计而非精确值。真实的 PyTorch 实现通常比这个更大。重点是**结构**：激活值与 $B$ 成正比，其中 $BHT^2$ 一项与序列长度成**平方**关系。
:::

那个 $BHT^2$ 项在 $T=1024$ 时是 26.2M，占单层的一半左右。$T$ 翻倍到 8192，它会膨胀 64 倍，而其他项只涨 8 倍——**这正是 FlashAttention 存在的理由**：它通过 tiling 让 $N \times N$ 的分数矩阵永远不落地到 HBM，把这一项从 $O(T^2)$ 压到 $O(T)$。（细节见 [FlashAttention 那篇](../inference/flash-attention)）

### 最终表达式

$$
\text{Peak Memory} \approx 26.25 + 9.76 \cdot B \quad \text{(GB)}
$$

## 三、80GB 的 H100 能放多大的 batch？

**训练**（含 AdamW 状态）：

$$
26.25 + 9.76B \le 80 \implies B \le 5.50 \implies B_{max} = 5
$$

**推理**（只需参数）：

$$
6.56 + 9.76B \le 80 \implies B \le 7.52 \implies B_{max} = 7
$$

::: note 推理这个数其实严重低估了
推理没有 backward，**不需要保存任何中间激活**——算完一层就可以丢掉，只留下一层的工作缓冲区。

正因为层与层之间只传一个 $B \times T \times D$ 的张量、不需要回头，推理天然适合 **Pipeline Parallelism**：把 48 层切成若干段放在不同 GPU 上，前一段算完把激活传给下一段就行，通信量极小（相比 tensor parallel 每层都要 all-reduce）。训练时 pipeline 的麻烦在于 backward 要反向走一遍造成流水线气泡（bubble），推理则没有这个问题——所以大模型多机部署常见的形态就是"节点内 tensor parallel + 跨节点 pipeline parallel"。

所以推理的显存瓶颈根本不是激活值，而是 **KV Cache**：

$$
\text{KV Cache} = 2 \times L \times B \times T \times D \times \text{bytes} = 2 \times 48 \times 1024 \times 1600 \times 2\,\text{(fp16)} = 0.31\,\text{GB} \cdot B
$$

按这个算，80GB 卡上推理能跑到 $B \approx 230$。这个 30 倍的差距，正是 vLLM / PagedAttention 那一整套工作要精细管理的对象。
:::

## 四、FLOPs：6ND 法则的由来

### 一个 transformer block 的前向传播（B=1）

| 操作 | FLOPs | 数值 |
|---|---|---|
| Q、K、V 投影（3×） | $6BTD^2$ | 15.73 GFLOPs |
| 注意力分数 $QK^\top$ | $2BT^2D$ | 3.36 GFLOPs |
| 注意力加权求和 $PV$ | $2BT^2D$ | 3.36 GFLOPs |
| 输出投影 | $2BTD^2$ | 5.24 GFLOPs |
| SwiGLU（$W_1, W_2, W_3$） | $6BTD \cdot d_{ff}$ | 42.15 GFLOPs |
| **单层合计** | | **69.84 GFLOPs** |

::: tip 一个矩阵乘 = 2mnk
所有这些 $2 \times$ 的系数都来自同一件事：$m \times k$ 乘 $k \times n$ 的矩阵乘法需要 $mnk$ 次乘法和 $mnk$ 次加法，共 $2mnk$ FLOPs。上面每一行都能用 size 和这个关系推出来。
:::

值得注意：**SwiGLU 占了单层的 60%**（42.15 / 69.84），而注意力的两个 $T^2$ 项加起来只占 10%。在 $T = 1024$ 这个长度上，Transformer 的算力其实主要花在 FFN 上，"attention 是瓶颈"只在长上下文时才成立（$T > D$ 时 $T^2 D$ 才开始压过 $TD^2$，本例中的临界点是 $T = 1600$）。

### 一个训练 step 的总 FLOPs（B=1）

| 阶段 | FLOPs | TFLOPs |
|---|---|---|
| 前向（48 层 + LM head） | 3,516,769,894,400 | 3.52 |
| 反向（≈ 2× 前向） | 7,033,539,788,800 | 7.03 |
| **前向 + 反向** | **10,550,309,683,200** | **10.55** |
| AdamW 更新 | 24,606,792,000 | 0.025 |

::: note 为什么 backward ≈ 2× forward
对于 $Y = XW$，前向算一次矩阵乘。反向要算**两个**梯度：$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top$（传给上一层）和 $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$（更新权重）。两次矩阵乘，规模与前向相同，所以是 2×。
:::

AdamW 的参数更新（每个参数约 15 次逐元素运算）只占前向+反向的 **0.23%**，完全可以忽略。

但要注意：这一步虽然 FLOPs 可忽略，却是**纯 memory-bound** 的。所谓"memory-bound"，是指瓶颈不在算力而在显存带宽——GPU 的 ALU 在空转等数据。AdamW 更新每一个参数时要碰四个张量：

| 操作 | 张量 | 流量 |
|---|---|---|
| 读 | 参数 $\theta$、梯度 $g$、动量 $m$、方差 $v$ | $4P$ |
| 写 | 参数 $\theta$、动量 $m$、方差 $v$ | $3P$ |
| **合计** | | **$7P$ 次元素访问 ≈ 4 个参数量大小的张量往返** |

fp32 下 $P = 1.64\text{B}$，就是 $7 \times 1.64\text{B} \times 4\,\text{bytes} \approx 46\ \text{GB}$ 的显存流量，却只做了 246 亿次 FLOP。**算术强度（arithmetic intensity）约 0.5 FLOP/byte**，而 H100 想跑满算力需要约 300 FLOP/byte——差了近三个数量级。在 H100 的 3.35 TB/s 带宽下，光搬这 46 GB 就要 ~14 ms；而它对应的 0.025 TFLOPs 计算量，理论上只要 0.1 ms。

所以在真实的 profile 里，optimizer step 的耗时占比会远高于 0.23%。这正是 **fused optimizer kernel**（把多次逐元素操作合并成一个 kernel，只读写一遍显存）存在的意义。

### 与 6ND 法则对照

$$
\text{FLOPs per step} \approx 10.55 \cdot B \ \text{(TFLOPs)}
$$

业界常用的经验公式是 **训练总 FLOPs ≈ 6ND**（$N$ 是参数量，$D$ 是 token 数）。验证一下：$B=1$ 时一个 step 处理 1024 个 token，

$$
6 \times 1.64 \times 10^9 \times 1024 = 1.01 \times 10^{13} = 10.1\ \text{TFLOPs}
$$

和我们逐项算出的 10.55 只差 4%。那多出来的部分就是注意力的 $T^2$ 项——**6ND 法则默认忽略注意力，这在 $T \ll D$ 时成立，长上下文时会低估**。

## 五、单卡 H100 训练要多久？

| 参数 | 取值 |
|---|---|
| H100 峰值算力（TF32） | 495 TFLOP/s |
| MFU（模型算力利用率） | 50% |
| 有效吞吐 | 247.5 TFLOP/s |
| Batch size | 1,024 |
| 训练步数 | 400,000 |

$$
\begin{aligned}
\text{每步 FLOPs} &= 10.55 \times 1024 = 10{,}804\ \text{TFLOPs} \\
\text{总 FLOPs} &= 400{,}000 \times 10{,}804 = 4.32 \times 10^{21} \\
\text{时间} &= \frac{4.32 \times 10^{21}}{247.5 \times 10^{12}} = 1.75 \times 10^7\ \text{秒} \\
&\approx 4{,}850\ \text{小时} \approx \mathbf{202\ 天} \approx 6.7\ \text{个月}
\end{aligned}
$$

总共处理 $1024 \times 1024 \times 400{,}000 \approx$ **419B tokens**。

::: warning 关于那个 495 TFLOP/s
H100 SXM 的 TF32 Tensor Core 算力，**稠密**是 ~495 TFLOP/s，官方 spec sheet 上标的更大的数字是开了 2:4 结构化稀疏的。实际 LLM 训练用不上稀疏，所以稠密值是对的口径。另外真实训练几乎都用 BF16（~990 TFLOP/s），会快一倍。

**MFU（Model FLOPs Utilization）** 指有效算力占峰值的比例。50% 是一个相当乐观的假设——GPT-3 论文级别的大规模训练通常在 35~45%，小模型或者流水线气泡多的情况能掉到 20% 以下。
:::

::: tip 419B tokens 合理吗？
按 **Chinchilla 定律**（$D \approx 20N$），1.64B 参数的模型"计算最优"的训练量是约 **33B tokens**。419B 是它的 12.7 倍——这是典型的**过训练（over-training）**。

现代模型（Llama 3、Qwen 3 等）都远超 Chinchilla 比例，因为 Chinchilla 优化的是"给定训练预算下的最佳 loss"，而工业界真正在意的是**推理成本**：一个小而过训练的模型，部署时永远比大而欠训练的模型便宜。训练一次，推理十亿次。
:::

## 六、单卡只能放 B=5，怎么实现 B=1024？

这是第三节和第五节之间那个刺眼的矛盾：我们算出单卡最多放 5，却假设 batch size 是 1024。

### 方案一：Gradient Accumulation（梯度累积）

用时间换空间。梯度对 batch 是**线性可加**的，所以可以分多次前向+反向，累积梯度后再更新一次：

```
effective_batch     = 1024
micro_batch         = 5
accumulation_steps  = ⌈1024 / 5⌉ = 205
```

每个训练 step：

1. 循环 205 次：forward(micro_batch=5) → loss → backward（梯度**累加**，不清零）
2. 累积完成后执行一次 `optimizer.step()`
3. `optimizer.zero_grad()`

数学上完全等价于一次性处理 $B=1024$。**总 FLOPs 不变**，所以单卡仍然是那 202 天——梯度累积解决的是"放不下"，不是"跑得慢"。

::: warning 一个常见的坑
如果 loss 用的是 `mean` 归约，每个 micro-batch 的 loss 要除以 `accumulation_steps` 再 backward，否则累积出来的梯度会是正确值的 205 倍。另外 205 × 5 = 1025 ≠ 1024，实际工程中会把 micro_batch 选成能整除的数（比如 4，256 步）。
:::

### 方案二：Data Parallelism（数据并行）

每张 GPU 持有**完整的模型副本**，处理**不同的数据子集**：各自 forward + backward → All-Reduce 同步梯度取平均 → 各自 `optimizer.step()`。

| GPU 数量 | 每卡 batch size | 预计训练时间 |
|---|---|---|
| 1 | 5（+ 梯度累积） | ~202 天 |
| 8 | 128 | ~25 天 |
| 32 | 32 | ~6 天 |
| 64 | 16 | ~3 天 |

注意这张表是**理想线性加速**。实际上每步都要 All-Reduce 全部 1.64B 个梯度（fp32 下 6.56 GB），卡越多通信占比越高。节点内有 NVLink（~900 GB/s）还好，跨节点走 InfiniBand 就会成为瓶颈——这也是 [NVSwitch / NVL72 那套互联架构](../inference/nvlink-nvswitch-topology) 存在的理由。

而且要注意：**纯数据并行并不省显存**，每张卡还是要扛完整的 26.25 GB 静态开销。这就引出了下面两个方向。

### 方案三：ZeRO / FSDP —— 把静态开销也切开

既然 16 bytes/param 里有 12 bytes 是梯度和优化器状态，那就把它们也沿 GPU 切分：

| 阶段 | 切分内容 | N 卡下每卡显存 |
|---|---|---|
| ZeRO-1 | 优化器状态 | $4P + 4P + 8P/N$ |
| ZeRO-2 | + 梯度 | $4P + (4P + 8P)/N$ |
| ZeRO-3 / FSDP | + 参数 | $16P/N$ |

ZeRO-3 下 8 张卡，静态开销从每卡 26.25 GB 降到 3.3 GB。代价是参数需要在用到时临时 all-gather 回来，通信量显著增加。

### 方案四：Activation Checkpointing —— 把激活值也换掉

激活值那 9.76·B GB 同样可以用时间换：只保存每层的输入（不保存所有的 activation），反向时重新前向一遍算出中间结果。显存从 $O(L)$ 降到 $O(\sqrt{L})$ 甚至 $O(1)$，代价是多一次前向——总 FLOPs 从 3× 变成 4×（约 +33% 时间）。

这是超长上下文训练的必备手段。

### 方案五：Model Parallelism —— 模型本身放不下时

| 策略 | 做法 |
|---|---|
| **Tensor Parallelism** | 把单个矩阵乘切分到多张 GPU（如按 $d_{ff}$ 维度切 FFN），每层都需要通信，通常只在节点内 NVLink 上做 |
| **Pipeline Parallelism** | 不同层放在不同 GPU 上，流水线处理，通信量小但有 bubble |

::: note
真实的大规模训练是这些策略的**组合**——所谓 3D/4D 并行：tensor parallel（节点内）× pipeline parallel（跨节点）× data parallel（跨副本）+ ZeRO + 梯度累积 + activation checkpointing。这是 CS336 后半程分布式训练部分的主题。
:::

## 七、可以带走的几个数字

如果只记三件事：

1. **16 bytes/param** —— fp32 + AdamW 的静态显存。想快速判断一个模型能不能训，先乘 16。
2. **6ND** —— 训练总 FLOPs。想快速估算训练成本，参数量 × token 数 × 6，再除以 (峰值算力 × MFU)。
3. **激活值 ∝ B，静态开销与 B 无关** —— 所以 batch size 只能在剩余显存里挤，而"剩余"往往少得可怜。

这三个公式加起来，你就能在餐巾纸背面回答"训一个 X B 的模型要几张卡、几天、多少钱"——而这恰恰是当下 AI 行业里最常被问、又最少被真正算过的问题。
