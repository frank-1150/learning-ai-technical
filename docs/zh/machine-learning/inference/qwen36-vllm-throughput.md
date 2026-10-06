---
date: "2026-10-05"
meta-description: 单卡 H20 上通过 prefix cache、FP8、每步 token 预算与 max_num_seqs 调优，将 Qwen3.6-35B-A3B 的 vLLM 服务吞吐从 3.9 提升到 20.6 req/s
title: "单卡 H20 上把 Qwen3.6-35B-A3B 的服务吞吐提到 5.3 倍"
description: 一次从 workload、roofline 与 kernel trace 出发的 vLLM 推理优化复盘
tags: [vllm, inference, qwen, moe, fp8, prefix-cache, cuda, performance]
---

# 单卡 H20 上把 Qwen3.6-35B-A3B 的服务吞吐提到 5.3 倍：一次 vLLM 推理优化复盘

> **一句话结论**：同一张 H20、同一套真实长度分布的负载下，服务的最大吞吐从 3.9 req/s 提到 20.6 req/s（×5.3），只用了四步：开 prefix cache、换 FP8 权重、调大每步 token 预算、调大 max_num_seqs。后三步其实都在解决同一个问题——**每一步喂给 GPU 的 token 太少，CPU 发射 kernel 的速度跟不上 GPU**。最终达到 roofline 理论上限的 54%。

这篇文章按实验顺序复盘：每一步看到了什么现象、据此判断瓶颈在哪、为什么做下一步，以及怎么算"离上限还有多远"。文中标 **🔁 待复测** 的地方，是数据不太符合预期、或者证据还不够硬的结论，最后一节汇总成了复测清单。

**适合谁读**：用 vLLM 部署 MoE 或混合架构（线性注意力 + 全注意力）模型、想搞清楚"吞吐卡在哪"的工程师。需要对 prefill / decode、continuous batching、CUDA kernel 发射有基本了解。

# 背景：模型、负载和测法

## 模型与硬件

| 项目 | 说明 |
|-|-|
| 模型 | Qwen3.6-35B-A3B：总参数约 35B，每个 token 激活约 3B。40 层里 30 层是 Gated DeltaNet 线性注意力（GDN），10 层是全注意力；FFN 是 256 个专家、每 token 选 8 个，外加一个共享专家。 |
| 量化 | 基线用 AWQ（W4A16，权重 4bit、计算 FP16）；优化后用官方 FP8 权重（W8A8，线性层用 FP8 tensor core）。 |
| 硬件 | 单卡 H20：FP16 稠密算力约 148 TFLOPS，FP8 约 296 TFLOPS，HBM 带宽约 4 TB/s。 |
| 推理框架 | vLLM 0.25.1，默认开启 chunked prefill 和 torch.compile + CUDA graph（CUDA graph 只覆盖 ≤256 token 的步）。 |

## 负载特征

这是一个"读长文、写短 JSON"的摘要类任务：输入约 3,200 token（p50 约 3,150），其中开头约 2,570 token 是所有请求完全相同的系统提示词；输出很短，p50 约 73 token。这个形态决定了后面几乎所有判断：

- **prefill 主导**：每条请求的计算量里，prefill 远大于 decode。
- **共享前缀很长**：不复用就是在反复重算同一段文字。
- **decode 很短**：一条请求在 batch 里只待约 73 步就走了，调度器需要不停补新请求进来。

## 怎么测

所有步骤用同一个自写的开环压测工具复测，保证可比：

- **请求内容**：token ID 是随机生成的（不含任何真实数据），但长度按真实样本的（用户输入长度，输出长度）成对抽样；所有请求共享同一段 2,570 token 的固定前缀，复现真实的前缀命中率（实测 66%，与线上一致）。输出用 `ignore_eos` + `min_tokens` 固定长度。
- **到达方式**：泊松到达，而不是固定并发的闭环压测。后面会看到，这个差别直接改变了瓶颈在哪（卡在 max_num_batched_tokens 还是 max_num_seqs）。
- **最大吞吐**：以 30 或 60 req/s 过载打 60 秒，取 3 次重复的均值；中等负载（6 req/s）下记录 TTFT / TPOT。
- **预热**：每次重启后先打 3 轮预热再计数。这一条踩过坑，见"踩过的坑"一节。
- **profiler**：kernel 级分析用 vLLM 内置的 torch profiler 单独采集，吞吐数字只取自不开 profiler 的运行。

# 总览：四步走到 5.3 倍

![图 1：每一步的实测最大吞吐（柱）与三种理论上限（线）。颜色区分量化方式：橙色 AWQ，蓝色 FP8](/machine-learning/inference/qwen36-vllm-throughput/zh/fig01_roadmap.png)

| 步骤 | 最大吞吐 | 达 roofline 上限 | 关键观察 |
|-|-|-|-|
| **S0 基线**  <br/>AWQ，无 prefix cache，预算 2048，seqs 128 | 3.9 req/s | 46%（上限 8.5） | 约 2/3 的 FLOPs 花在重算相同的系统提示词上；日志显示 KV cache 命中率为 0。 |
| **S1 + prefix cache** | 9.9 req/s（×2.5） | 45%（上限 21.9） | 前缀命中 66%；AWQ 下 GPU 基本跑满，受算力限制。 |
| **S2 + FP8 权重** | 12.0 req/s（×3.1） | 35%（上限 33.7） | kernel 快了 1.7 倍，端到端只快 21%：GPU 一大半时间在等 CPU。 |
| **S3 + 调大预算**  <br/>max-num-batched-tokens 2048 → 16384 | 15.9 req/s（×4.1） | 47%（上限 33.7） | 每步 token 变多，但真实负载下被 max_num_seqs 封顶在约 1,750。 |
| **S4 + 调大 max_num_seqs**  <br/>128 → 384 | 20.6 req/s（×5.3） | 54%（上限 38.2） | 每步约 10,000 token，GPU 空闲降到 1.7%。 |

下面这张图把"看到什么 → 于是做什么"的决策链串起来，后面每一节展开一步。

![图 2：决策链](/machine-learning/inference/qwen36-vllm-throughput/zh/fig02_decision_chain.png)

# 第一步：开 prefix cache（3.9 → 9.9 req/s）

## 为什么先做它

基线下每条请求要 prefill 约 3,200 token，其中 2,570 token 是一模一样的系统提示词。按 FLOPs 估算，不复用时每条请求约 16.8 TFLOP；如果共享前缀的 KV 和状态能直接复用，降到约 6.1 TFLOP。这是一个**不损失精度、直接砍掉 2/3 计算量**的改动，理论上限从 8.5 提高到 21.9 req/s，所以排第一。

## 混合模型架构（GDN + attention）的坑：块大小是 1056，不是 16

普通 Transformer 开 prefix cache 只需要 `--enable-prefix-caching`。这个模型有 30 层 GDN 线性注意力，

Gated DeltaNet 保存的 prefix cache 是一个递归状态 $S_t$，每一步用当前 token 更新。这个状态的大小固定为 $d \times d$。

GDN 没有逐 token 的 KV，而是每条序列维护一个固定大小的递归状态（每层约 2.1 MB，FP32）。要缓存"前缀到某处为止"的结果，就得在那个位置存一份状态快照。vLLM 的做法是：

- 全注意力层的 KV 页和 GDN 的状态页共用一个页大小，所以注意力页必须至少装得下一份 GDN 状态，**块大小被撑到 1,056 token**。
- vLLM 启动的参数需要加 `--mamba-cache-mode align`：只在块边界处保存状态，prefill 的切块（chunked prefill 的 chunk size）也被强制对齐到 1,056 的整数倍。

结果是前缀只能按整块命中：2,570 token 的共享前缀只命中前 2 块（2,112 token），剩下 458 token 每次仍要重算。命中率 66%(2112/3200)，与线上实测一致。

> **可以白捡的优化**：如果把系统提示词的共享部分设计成正好对齐 1,056 的整数倍（比如精简到 2,112 token，或把稳定内容补到 3,168 token），就能让整段前缀都命中。本文没有做这个实验。

## 结果

- 最大吞吐 3.9 → 9.9 req/s（×2.54）。
- 用真实样本回放，端到端延迟 p50 2,982 → 1,545 ms，首 token 延迟 p50 1,449 → 699 ms。
- 代价：块变大后可用 KV 容量少了约 7.9%。
- 效果：离线用 100 条真实样本、4 个评估维度对比，3 个维度差异在噪声内，1 个维度下降但多重比较校正后不显著。

> **已补测**：之后又在两个 prompt 上各用 500 条线上样本、带"同配置重跑"作为噪声参照复测了一遍，AWQ + prefix cache 在所有维度上都与重跑重合，见"效果验证"一节。

# 第二步：换 FP8 权重（9.9 → 12.0 req/s）

## 为什么做它

开了 prefix cache 之后，AWQ 版本 GPU 几乎一直在算（kernel 间空闲约 2%），是典型的算力受限。计算量的大头是线性层（注意力投影 + MoE 专家），而 H20 的 FP8 算力是 FP16 的 2 倍。AWQ 虽然权重是 4bit，但计算走 FP16；换成 W8A8 的 FP8，线性层可以利用 FP8 tensor core。按 FLOPs 拆分估算，理论上能快约 1.8 倍。

## W8A8 到底在哪里量化：一次计算的精度流向

"W8A8"指线性层的权重（W）和激活值（A）都是 8 位浮点（FP8 E4M3，最大约 ±448），其余计算保持 BF16 / FP32。我们用的[官方 FP8 权重](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)采用分块缩放：权重按 128×128 的块、激活按"每个 token 每 128 个通道"一组，各自配一个 FP32 缩放系数（配置里是 `weight_block_size: [128, 128]`、`activation_scheme: dynamic`）。

![图 2b：W8A8 推理的精度流向。上：一个 FP8 线性层内部；下：一个 decoder 层里每一步用什么精度](/machine-learning/inference/qwen36-vllm-throughput/zh/fig02b_w8a8_precision_flow.png)

**一个 FP8 线性层（Y = X · W）的计算过程**：

1. **权重：离线量化，运行时不转换。** 做权重时就把每个 128×128 块除以缩放系数 `s_w = max|W_块| / 448`，量化后的权重存成 FP8，`s_w` 用 FP32 存。
2. **输入 X：每次前向都现场量化（BF16 → FP8，向下）。** 每个 token 每 128 个通道求一次最大绝对值，`s_x = max / 448`，再把 `X / s_x` 舍入成 FP8。这一步是一个独立的小 kernel，就是 profiling 里约 5% 的"激活量化"开销。
3. **矩阵乘：FP8 × FP8 走 Tensor Core**，H20 上吞吐是 FP16 的 2 倍。
4. **累加：FP32（向上）。** 每算完一个 128 宽的 K 分块，把部分和乘上 `s_x · s_w` 还原量级，再累加进 FP32 累加器。
5. **输出：FP32 → BF16（向下）**，交给下一步。

按 128 一组缩放，是因为 E4M3 只有 3 位尾数：整张矩阵只用一个缩放系数时，一个离群值就会把其他值都压成 0；分组后误差只影响局部。

| 类别 | Qwen3.6-35B-A3B 里具体是哪些 |
|-|-|
| **量化成 FP8**（权重 + 激活） | 全注意力的 q / k / v / o 投影；GDN 的 in_proj_qkvz 和 out_proj；MoE 256 个专家和共享专家的 gate_up / down。也就是绝大部分参数和 FLOPs。 |
| **不量化**（保持 BF16） | 词嵌入、lm_head、所有 RMSNorm、路由 gate、共享专家门控、GDN 的 in_proj_a / b、conv1d、A_log、dt_bias；注意力计算本身、激活函数、残差相加；KV cache（`kv_cache_dtype=auto`，即 BF16）。 |
| **向上转换**（升到 FP32） | 每个 FP8 矩阵乘的累加；RMSNorm 内部计算；注意力的 softmax 与累加；路由的 softmax / top-8；GDN 递推状态（`mamba_ssm_dtype: float32`，也是 prefix cache 存的状态快照）；采样前的 logits。 |
| **向下转换**（降低精度） | 每个 FP8 线性层之前的激活量化 BF16 → FP8：每层要做多次（GDN 层 2 次、注意力层 2 次、每个被选中的专家和共享专家各 2 次）；每个矩阵乘的输出 FP32 → BF16。 |

这也解释了后面看到的两个现象：只有矩阵乘部分享受 FP8 的 2 倍算力，注意力和 GDN 的计算不变；每个 FP8 线性层前多出一个量化小 kernel，每步 token 少时这些额外的 kernel 发射会放大 CPU 侧的瓶颈。另外，prefix cache 缓存的是 BF16 的 KV 和 FP32 的 GDN 状态，并不是 FP8。

## 结果不如预期：只快了 21%

实测只从 9.9 提到 12.0 req/s。用 profiler 抓 kernel 时间线，结论很清楚：

- **GPU 计算本身达到了预期**：FP8 的 kernel 总时间比 AWQ 少 41%（1.70 倍），MoE 快 2.3 倍，稠密 GEMM 快 1.7 倍；注意力和 GDN 仍是 BF16，耗时不变；新增的激活量化 kernel 约占 5%。
- **但 GPU 只有约 40% 的时间在算**（AWQ 约 90%）。FP8 每个 kernel 之前平均空闲约 42 µs（AWQ 约 6 µs），而且空闲几乎全是大量亚毫秒的小间隙——这是典型的 **launch-bound**：GPU 算完一个 kernel，下一个还没被 CPU 发射过来。

> **Profiling 时间线**
>
> ![FP8 模型的 Perfetto trace，GPU stream 中存在大量 kernel 间空隙](/machine-learning/inference/qwen36-vllm-throughput/shared/trace_fp8_perfetto.png)
>
> FP8的 profiling
>
> ![AWQ INT4 模型的 Perfetto trace，GPU stream 更连续](/machine-learning/inference/qwen36-vllm-throughput/shared/trace_awq_perfetto.png)
>
> Int4 profiling，可以看到，FP8 比 INT4 的 GPU stream 空洞还是多不少。具体也体现在
>
> **怎么打开**
>
> 1. 推荐用 Perfetto：浏览器打开 https://ui.perfetto.dev，点左侧 "Open trace file"，直接选 `.json.gz`，不用解压。文件在浏览器本地解析，不会上传。解压后约 300 MB，第一次加载要几十秒。
> 2. 也可以用 `chrome://tracing`，老一点，文件大时比较卡。
> 3. 或者用 TensorBoard 的 PyTorch Profiler 插件，要多装一些依赖，不推荐。
>
> **在 Perfetto 里怎么看出"GPU 在等 CPU"**
>
> 1. 找到 GPU stream 那一行（kernel 都在这里），以及 CPU 的主线程那一行（`execute_context_...` 这些标注都在这里）。
> 2. 选中一个 `execute_context_1(1056)_generation_N` 标注，按 `F` 放大到这一步。
> 3. 在 FP8 @2048 的 trace 里，GPU 那一行的 kernel 之间有大量细小空隙。点任意一个 kernel，会有一条箭头连回 CPU 上对应的 `cudaLaunchKernel`，能看到 GPU 几乎是"刚发射就执行"，然后等下一个。
> 4. 换成 `fp8_b4096` 的 trace 看同样位置，GPU 那一行基本连成一片，CPU 的发射明显跑在 GPU 前面。

- **为什么 FP8 更容易 launch-bound，两个原因**：

  1. 因为 FP8 精度计算时所需要的计算量少了、时间更短了，所以 GPU kernel 所占的时间更短，CPU 侧的固定开销占比就更高。
  2. FP8 精度模型和 AWQ INT4 模型使用的 kernel 不同，导致的 kernel launch 开支不同：Triton 的 fused MoE、DeepGEMM、激活量化这些 kernel 都经 Python 层发射，单次开销远大于 AWQ 用的 Marlin / cuBLAS 这类 C++ 算子。
- **为什么小的 step 会导致 GPU 利用率不高？** GPU 利用率的计算公式是由 GPU 计算的时间除以整个任务的时间。那么上面提到了，如果是 launch bound 的话，就会导致 GPU 空闲等待 CPU 提交 kernel。GPU 运行 kernel 的时间和 kernel 里面的工作量有关。kernel 里面的工作量不大的话，也就是我们上面说的一个 step 的 batch size 不大，一个 step 里面处理的 token 量不够大的话，就会导致 GPU 运行这个 kernel 时间比较短，CPU 来不及准备第二个 kernel 就需要 GPU 进行等待。在理想情况下，GPU 一直忙碌，CPU 准备任务的时间的延迟就会被隐藏在 GPU 计算的延迟里面，这叫 **latency hiding（延迟隐藏）。**
- **为什么每步这么"小"（这里的 step 指的是 Continuous batching 里面的 scheduler step）**：默认的 max-num-batched-tokens 是 2048，而 align 模式要求 prefill 切块是 1,056 的整数倍，所以每步只能放下 1 块 prefill（1,056 token）加若干 decode。每步约 1,000 token 的计算量太小，填不满 CPU 发射一整步 kernel 的时间；这个步长也超出了 CUDA graph 的捕获范围（≤256），只能 eager 逐个发射。

于是瓶颈从"GPU 算不过来"变成了"CPU 喂不过来"。下一步的方向很自然：**让每一步的计算量变大**，把固定的发射开销摊薄。

> **什么是 step？**
>
>  Step 解决的主要问题是：在现代推理引擎出现之前，多个推理请求会被包装成a batch of sequence，丢进自回归生成进行生成。但是，不同的自回归序列生成的长度是不一样的，有些会提前终止。这些提前终止的序列被捆绑在一个 batch 里面进行并行计算，从而导致了额外的算力浪费。
>
> 早期 ORCA 将这种机制称为 **iteration-level scheduling**：scheduler 在每个 iteration 重新选择要执行的请求，只让 execution engine 前进一个 iteration，然后再次进行调度。这样，已经结束的请求可以立即退出，新到达的请求也可以加入下一轮 batch，这正是 Continuous Batching 的核心思想。
>
> [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)

![Orca 论文中 iteration-level scheduling 的连续批处理示意图](/machine-learning/inference/qwen36-vllm-throughput/shared/ref_orca_continuous_batching.png)

*图源：Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf), OSDI 2022, Figure 4。*

# 第三步：调大每步 token 预算（12.0 → 15.9 req/s）

## 先在合成负载上找拐点

为了看清"每步多少 token 才能喂饱 GPU"，我先用一个纯 prefill 的合成负载（每条请求前缀都不同、0 命中，输入 4,096 + 256、输出 32，128 并发闭环）扫了 max-num-batched-tokens = 2048 / 4096 / 8192 / 16384 / 32768，每档抓 profiler 统计每步实际 token 数、kernel 间空闲和 kernel 达到的算力。

![图 3：GPU kernel 间空闲占比 vs 每步 token 数。FP8 在约 3,000 token/步处出现拐点](/machine-learning/inference/qwen36-vllm-throughput/zh/fig03_idle_vs_tokens.png)

![图 4：kernel 达到的算力 vs 每步 token 数。FP8 的 GEMM + MoE kernel 在大 batch 下达到约 224–230 TFLOPS](/machine-learning/inference/qwen36-vllm-throughput/zh/fig04_tflops_vs_tokens.png)

- **拐点在每步约 3,000 token**：FP8 的 GPU 空闲占比从 1,100 token/步时的 64% 降到 3,290 token/步时的 6.5%，再往后接近 0。AWQ 本来就不太空闲，变化小得多。
- **FP8 相对 AWQ 的吞吐比**从预算 2048 时的 0.83 倍（反而更慢）跳到 4096 时的 1.81 倍，之后稳定在约 1.8 倍，正好是理论预期。
- **机制验证**：以 FP8 模型为例，看每个 kernel 从被 CPU 发射到在 GPU 上开始执行的等待时间。预算 2048 时中位数只有 9 µs、72% 的 kernel 在 50 µs 内开始——CPU 一发射 GPU 就执行，说明 GPU 在等 CPU；到 4096 时中位数变成 4.6 ms，CPU 已经领先 GPU 很多，GPU 不再挨饿。

> **待复测**：图 4 中 FP8 的"端到端有效算力"（虚线）在 32768 档从约 158 掉到 104 TFLOPS，而 kernel 算力（实线）继续上升。推测是闭环压测下大 batch 让 prefill 和 decode 分批同步、请求节奏被打乱造成的，与 kernel 无关，但还没有用开环压测验证。

## 回到真实负载：提升了，但没到拐点

把预算从 2048 调到 16384 后，真实负载下的最大吞吐从 12.0 提到 15.9 req/s。但 profiler 显示每步只有约 1,750 token，FP8 仍有 37% 的时间空闲——离 3,000 的拐点还差得远。预算明明给到了 16384，为什么用不满？

原因在真实负载的形态：每条请求平均要做约 1,080 token 的新 prefill（命中前缀之外的部分），然后 decode 约 73 步。过载时 128 个并发槽位几乎全被正在 decode 的请求占着，只有一个请求生成结束腾出槽位（槽位指的是scheduler 一次 step 里的允许的同时在跑的请求总数，对应 max_num_seqs），调度器才能放新请求进来做 prefill。粗算每步只能新进约 1～2 条请求，每步 token 数 ≈ 1.5 × 1,080 + 128 条 decode ≈ 1,750，和实测吻合。**此时卡住每步 token 数的是 max_num_seqs，而不是 token 预算。**

> **合成负载和真实负载的瓶颈不一样**：在前面的闭环合成负载里，128 条请求同时开始 prefill，每步 token 数只受预算限制；换成泊松到达、短输出的真实负载，限制就换成了并发槽位数。调参结论一定要回到真实形态的负载上验证。

# 第四步：调大 max_num_seqs（15.9 → 20.6 req/s）

既然并发槽位是新瓶颈，就把 max_num_seqs 从 128 提到 256、384（预算固定 16384）。这个模型的 KV 缓存容量约 210 万 token，384 条 × 约 3,300 token ≈ 127 万，放得下。

![图 5：增大 max_num_seqs。左：最大吞吐；右：GPU 空闲占比，标注为每步 token 数](/machine-learning/inference/qwen36-vllm-throughput/zh/fig05_max_num_seqs.png)

| max_num_seqs | FP8 最大吞吐 | AWQ 最大吞吐 | FP8 / AWQ | FP8 每步 token / 空闲 | FP8 中负载 TTFT / TPOT p50 |
|-|-|-|-|-|-|
| 128 | 15.9 req/s | 10.8 req/s | 1.48× | 1,756 / 37% | 235 / 18.0 ms |
| 256 | 19.7 req/s | 11.9 req/s | 1.66× | 3,038 / 17% | 228 / 17.0 ms |
| 384 | 20.6 req/s | 12.5 req/s | 1.65× | 10,068 / 1.7% | 288 / 23.0 ms |
| 512 | 启动失败 | 12.8 req/s | — | — | — |

- **FP8 收益大**：每步 token 数越过 3,000 拐点后，空闲从 37% 降到 1.7%，吞吐 +29%。
- **AWQ 收益小**：AWQ 本来就是 GPU 跑满（空闲 2.3%），多塞请求只能靠更大的 batch 略微提高 kernel 效率，+16%。
- **256 → 384 边际递减**：此时 GPU 已基本不空闲，再往上就要靠 kernel 本身变快了。

上面的最大吞吐只是过载下的一个点。为了按延迟目标规划容量，我又在 128 / 256 / 384 三档下分别做了完整的到达速率扫描（不开 profiler，每档 3 次 × 60 秒）：

![图 5b：三档 max_num_seqs 的吞吐与首 token 延迟。上排：完成吞吐 vs 到达速率；下排：TTFT p99 vs 完成吞吐](/machine-learning/inference/qwen36-vllm-throughput/zh/fig05b_seqs_rate_sweep.png)

| 配置 | 饱和吞吐 | TTFT p99 < 1 s 时的容量 | TTFT p99 < 2 s 且 TPOT p50 < 100 ms 时的容量 | 6 req/s 时 TTFT / TPOT p50 |
|-|-|-|-|-|
| AWQ，seqs 128 | 10.8 | 7.6 | 7.6 | 229 / 20.1 ms |
| AWQ，seqs 256 | 11.9 | 7.6 | 9.8 | 230 / 20.1 ms |
| AWQ，seqs 384 | 12.3 | 7.6 | 9.8 | 221 / 18.8 ms |
| FP8，seqs 128 | 15.0 | 13.6 | 13.6 | 259 / 20.9 ms |
| FP8，seqs 256 | 19.3 | 17.6 | 13.6 | 258 / 20.7 ms |
| **FP8，seqs 384** | **20.3** | **17.7** | **15.4** | 229 / 17.8 ms |

单位均为 req/s。几点读法：

- **按延迟目标算，FP8 + 256/384 的可用容量约 17.6 req/s，是 AWQ 的 2.3 倍。** 饱和吞吐的差距（20.3 vs 12.3）在延迟约束下反而更大，因为 AWQ 的首 token 延迟在 8 req/s 左右就开始上升。
- **AWQ 调大 max_num_seqs 只是多扛高峰**：饱和吞吐从 10.8 提到 12.3，但 TTFT p99 < 1 s 的容量一直是 7.6 req/s——GPU 本来就跑满了，多放进来的请求只是在排队。
- **中等负载下延迟不受影响**：6 req/s 时三档的 TTFT / TPOT 基本一样。之前一次测量里 384 档 TPOT 高出 35%，这次没有复现，大概率是那次的波动。
- 这组数据和上表的单点测量不在同一批节点上，FP8 128 档的饱和值（15.0 vs 15.9）差异就来自节点（见"踩过的坑"第 6 条）。

> **待复测（2 项）**
> - FP8 在 max_num_seqs=512 时启动必失败：torch.compile 在 profile_run 阶段编译 GDN 状态索引相关的 Triton kernel 时报 `KeyError: 'cubin'`，同配置 AWQ 正常。需要定位是 vLLM / Inductor 的问题还是环境问题，或在新版本上重试。
> - AWQ 在 384 档 profiler 窗口里测到的每步 token 数（2,936）反而比 256 档（6,358）低，与吞吐单调上升矛盾。推测是 60 步的采样窗口太短、恰好落在请求进出的低谷，需要加长窗口或多次采样。

# 一条统一规律：GPU 吃不吃得饱，只看每步 token 数

把所有实验（合成负载的预算扫描、真实负载的预算档、真实负载的 max_num_seqs 档）画到同一张图上，横轴是每步实际 token 数，纵轴是 GPU 空闲占比：

![图 6：所有配置下的 GPU 空闲占比 vs 每步 token 数（对数轴）。不论调的是哪个参数，点都落在同一条曲线上](/machine-learning/inference/qwen36-vllm-throughput/zh/fig06_idle_universal.png)

不管调的是 token 预算还是 max_num_seqs、负载是合成的还是真实的，FP8 的点都落在同一条曲线上：**每步低于约 3,000 token 就会因为 CPU 发射跟不上而空闲，高于它就能跑满**。所以调参的目标可以直接表述为"让稳态下每步 token 数越过 3,000"，至于用哪个参数实现，取决于当前是哪个限制在起作用。

# 离上限还有多远：三种理论上限怎么算

只看吞吐涨了多少，不知道还剩多少空间。我为每一步估算了三种理论上限（图 1 中的三条线），单位都是 req/s。

## 算力上限

每条请求的 FLOPs 分两部分：

- **线性层(MoE + full attention linear+ GDN)**：每 token 约 4.87 GFLOP（按 kernel 形状反推的模型维度计算）。需要计算的 token = 未命中的 prefill token + 输出 token。

一个线性层的权重形状是 `[K, N]`，每个 token 过一遍要做 K×N 次乘加，也就是 **2·K·N FLOP**。把每个 token 实际经过的所有权重矩阵加起来：

| 部分 | 层数 | 每层权重（K×N） | 每层每 token 乘加数 |
|-|-|-|-|
| MoE | 40 | 选中的 8 个专家 × 3 个矩阵（gate / up / down，各 2048×512） | 25.2 M |
| | | 共享专家 3 × 2048×512 | 3.1 M |
| | | 路由 gate 2048×256 | 0.5 M |
| | | **小计** | **28.8 M** |
| 全注意力 | 10 | q / k / v / 输出门控投影 2048→9216 | 18.9 M |
| | | o_proj 4096→2048 | 8.4 M |
| | | **小计** | **27.3 M** |
| GDN | 30 | in_proj_qkvz 2048→12288 | 25.2 M |
| | | in_proj_ba 2048→64 | 0.13 M |
| | | out_proj 4096→2048 | 8.4 M |
| | | **小计** | **33.7 M** |

```
40×28.8M + 10×27.3M + 30×33.7M ≈ 1153M + 273M + 1011M ≈ 2.44 G 乘加
× 2 = 4.87 GFLOP / token
```

- **MoE 只算实际激活的部分**：每个 token 只经过 top-8 专家，其余 248 个不参与计算。2.44 G 乘加约等于 2.4B 个激活的线性层参数，这就是"A3B"的含义。
- **"按 kernel 形状反推"**：profiler trace 里能看到每个 GEMM 的 N、K，9216、12288、4096 这些维度是从 trace 读出、再和模型配置对上的。9216 = q 16 头×256 + 输出门控 4096 + k、v 各 2 头×256；12288 = q、k 各 2048 + v 4096 + z 4096。
- **命中 prefix cache 的 token 直接复用 KV 和 GDN 状态，线性层不用再算**，所以开 prefix cache 能省下大块 FLOP。

- **注意力打分**：只有 10 层全注意力，每 token 每个上下文位置约 `10 层 × 4 × 16 个 q 头 × 256 维 = 163,840 ≈ 16.4 万 FLOP / (token · 上下文位置)` ，乘以各自的上下文长度。

一个 query token 对上下文里的一个位置，每个 q 头要做：

- **QKᵀ**：两个 256 维向量的点积，256 次乘加 = 2×256 FLOP；
- **PV**：用这个打分给 256 维的 V 加权累加，同样 2×256 FLOP；
- 合计 **4×256 FLOP**。乘的是 q 头数 16 而不是 KV 头数：GQA 让多个 q 头共享一份 KV，省的是显存和带宽，打分仍要每个 q 头各算一遍。

AWQ 全部按 FP16 峰值 148 TFLOPS 计；FP8 的线性层按 296 TFLOPS、注意力和 GDN（仍是 BF16）按 148 TFLOPS 计。算力上限 = 1 ÷（每请求所需时间）。

**自检**：4.87 GFLOP/token ÷ 2 ≈ 24.4 亿参数，加上未计入的 embedding 和 lm_head（约 5 亿），合计约 29 亿，与模型名里的"A3B"（约 30 亿激活参数）吻合。

## 带宽上限

按 4 TB/s 计，每条请求的访存有三块：

- **权重**：AWQ 约 22 GiB、FP8 约 34 GiB。每一步都要完整读一遍，所以按每步的 prefill token 数和 decode batch 大小分摊到每条请求。batch 越大，每条请求分摊得越少——这也是调大 max_num_seqs 后带宽上限从 67 跳到 157 req/s 的原因。
- **KV 缓存**：每个上下文 token 在 10 层全注意力里约占 20 KB。decode 每步每条序列要读一遍整个上下文，约 66 MB。
- **GDN 状态**：每条序列 30 层合计约 64 MB，decode 每步读一次、写一次，约 129 MB。这是混合架构特有的访存项，比 KV 还大。

## roofline 上限：算力和访存所需的时间不能直接相加

同一个 kernel 里，GPU 一边算当前这块数据，一边异步把下一块搬进来（上文提到的 latency hiding），所以一个 kernel 的耗时接近算力时间和访存时间中较长的那个，而不是两者之和。我的口径是分阶段取较大值：**prefill 和 decode 各自取 max(算力时间, 访存时间)，再把两个阶段相加**。重叠得有多充分取决于实现，所以下表同时列出最乐观（完全重叠）和最悲观（直接相加）的算法作为区间：

| 口径（req/s） | S0 | S1 | S2 | S3 | S4 |
|-|-|-|-|-|-|
| 只看算力（完全重叠，乐观） | 8.8 | 24.2 | 44.7 | 44.7 | 44.7 |
| 只看带宽 | 59.7 | 76.5 | 54.9 | 66.6 | 157 |
| **分阶段 roofline（主口径）** | **8.5** | **21.9** | **33.7** | **33.7** | **38.2** |
| 直接相加（悲观） | 7.7 | 18.3 | 24.6 | 26.7 | 34.7 |
| 实测 | 3.9 | 9.9 | 12.0 | 15.9 | 20.6 |
| 实测 / 分阶段 roofline | 46% | 45% | 35% | 47% | 54% |

几点读法：

- **整体是算力受限**：每一步的带宽上限都远高于算力上限。单看 decode 阶段则是带宽受限（每步要搬权重、KV 和 GDN 状态，计算却很少），但这个负载的 decode 很短，占比小。
- **S2 的达成率反而下降**（45% → 35%）：FP8 把上限抬高了，实测没跟上，差距就是 CPU 发射空洞。S3、S4 把它补回来，达成率回到 54%。
- **剩下的 46% 在哪**：S4 时 GPU 已基本不空闲，差距主要在 kernel 自身效率——FP8 GEMM 实际约 215 TFLOPS（峰值的 73%），注意力、GDN、采样和量化等 kernel 的效率更低。下一步要从 kernel 入手。

# 到达速率扫描：吞吐与延迟曲线

前面的"最大吞吐"都是过载下测的。实际容量规划还需要知道：在不同到达速率下，吞吐能不能跟上、延迟怎么变。下面在 max_num_seqs=128、预算 16384 的配置下，从 2 req/s 扫到 20 req/s，每档 3 次 × 60 秒，全程不开 profiler。

![图 7：完成吞吐 vs 到达速率。贴着对角线表示来多少做完多少，变平表示饱和](/machine-learning/inference/qwen36-vllm-throughput/zh/fig07_throughput_vs_rate.png)

![图 8：TTFT 与 TPOT 随实际吞吐的变化（对数轴），实线 p50，虚线 p99](/machine-learning/inference/qwen36-vllm-throughput/zh/fig08_latency_vs_throughput.png)

- **饱和点**：AWQ 约 10.9 req/s，FP8 约 15.8 req/s，与前面单独过载测得的 10.8 和 15.9 一致。
- **饱和前延迟平稳，饱和后排队爆炸**：同样是 10 req/s，AWQ 已接近饱和，TTFT p50 547 ms、TPOT p50 103 ms；FP8 还很轻松，TTFT p50 285 ms、TPOT p50 41 ms。
- **低负载下两者差不多**：2 req/s 时 TTFT p50 约 130–140 ms，TPOT p50 FP8 4.8 ms、AWQ 5.3 ms。FP8 的权重更大（约 34 vs 22 GiB），但低负载下没有表现出更慢。
- **按延迟目标规划容量**：如果要求 TTFT p99 < 1 秒，FP8 大约能承接 14 req/s（p99 636 ms），AWQ 约 8 req/s（p99 769 ms），FP8 能多扛约 75% 的流量。

> **这一节第一次测时数据是错的**：最初每个速率档都顺手抓了一次 profiler，FP8 只在 12.8 req/s 就饱和了，低负载的延迟也偏高。排查发现是反复开关 profiler 导致 FP8 引擎持续变慢（见"踩过的坑"第 2 条），上图是全程不开 profiler 重测的结果。AWQ 那组当时也抓了 profiler，但 AWQ 是 GPU 受限，CPU 侧的额外开销被掩盖，饱和值与单独过载测量一致，所以沿用。

> **已补测**：max_num_seqs 256 / 384 下的速率扫描见第四步的图 5b。同一配置在不同节点上的饱和吞吐差异约 2–6%，上面 FP8 与 AWQ 的比例按 ±5% 理解即可。

# 效果验证：prefix cache 和 FP8 会不会让输出变差

吞吐提上去之前，先要确认输出质量没有变差。prefix cache 理论上不改变计算结果，但混合架构下要缓存和恢复 GDN 状态快照，实现上有出错的可能；FP8 则会实实在在地改变数值精度。所以我对这条链路上的两个 prompt 都做了 A/B：information summary（根据创作者信号生成检索词，本文压测的就是它），copy generation（从检索结果里选一个话题并写推送文案，同一个模型，输入约 4,600 token、输出约 110 token）。

## 怎么测

- **样本**：每个 prompt 从最近 7 天的线上调用里按语言分层抽 500 条，保留线上真实的接受 / 拒绝比例（约 92% 接受）。
- **实验组**：4 组用完全相同的 prompt 版本和采样参数，只换后端服务：base（AWQ、不开 prefix cache，与线上一致）、base 重跑（同一配置再生成一遍）、AWQ + prefix cache、FP8 + prefix cache。
- **为什么要 base 重跑**：两个 prompt 都带随机采样（temperature 0.6 和 1.0），同一配置跑两次输出本来就不一样。base 重跑和 base 的差距就是"随机波动"的参照，其他组的差距只有明显超出它才算真的变化。
- **评估**：LLM 裁判按维度打分（0/1 或四档）。copy generation 上除了格式、选题和文案质量外，专门加了 4 个维度：语言是否与用户一致、是否安全且没有人身攻击、与用户特征的相关度、与检索词的相关度。另外用一个"两两等价"裁判判断两份输出对下游是否等价。所有比较都是逐条配对，报告均值和 95% bootstrap 置信区间。
- **裁判限流**：两个 prompt 共约 2.5 万次裁判调用。一开始在评测平台上把生成和打分绑在一起并发跑，很快撞上裁判模型的每分钟请求上限，大量打分失败。改成先只跑生成（十几分钟），再用一个自适应限速脚本单独打分：遇到限流就降速 30% 并重试，2 分钟无报错就加速，最终稳定在每分钟约 250 次，只有 13 次限流。

![图 9：各组相对 base 的配对差值（均值与 95% 置信区间）。灰色是同配置重跑，代表随机波动的范围](/machine-learning/inference/qwen36-vllm-throughput/zh/fig09_quality_forest.png)

## 结论

- **AWQ + prefix cache：没有任何可测出的变化。** 两个 prompt 的所有维度、接受率和等价率都和同配置重跑重合。
- **FP8 + prefix cache：质量分数没有下降，但输出变化略多。** information summary 上 FP8 输出与 base 的等价率比随机波动低约 6 个百分点（0.656 vs 0.716），说明数值精度确实让一部分输出走向了别的候选，但各项质量分数都没有变差。
- **唯一的业务影响：copy generation 上 FP8 的接受率低约 4 个百分点**（0.870 vs base 0.908、线上 0.914），多出来的主要是"没有相关候选"的拒绝。这些样本本身是边界情况：裁判认为 FP8 拒绝得对的只有 12%，而 base 当初接受得对的是 0%，也就是两边怎么判都不太好。单项检验 p = 0.018，多重比较校正后约 0.11，偏向真实但还不够确定。
- **看安全分时要小心**：information summary 上三组的安全分相对 base 都低约 0.05，连同配置重跑也一样，说明是 base 这次碰巧偏高。拿 FP8 直接和重跑比，差值只有 +0.004。如果没有重跑这一组，很容易误判成"FP8 降低了安全性"。

> **待复测**：copy generation 上 FP8 接受率低约 4 个百分点，需要再生成一组 FP8 重跑确认能否复现。另外，所有组（包括线上输出）的绝对分数都偏低（如选题质量约 0.33），说明裁判偏严或 prompt 有普遍问题，不影响 A/B 结论，但值得单独排查。

# 踩过的坑

1. **不预热会得出完全相反的结论**：第一次对比 FP8 和 AWQ，FP8 比 AWQ 慢 46%。原因是 FP8 路径的 Triton kernel 在第一次遇到新形状时要 JIT 编译，这部分时间被算进了压测。预热之后两者打平。之后每次重启都先打 3 轮预热。
2. **反复开关 profiler 会让引擎持续变慢**：在同一个服务上连续开关 5 次 torch profiler 后，FP8 的过载吞吐从 15.1 降到 13.3 req/s（−12%），TPOT p50 从 115 升到 131 ms，而且之后一直不恢复；这期间容器没有 CPU 限流，宿主机也还有空闲 CPU。只开关一次看不出影响（19.55 vs 19.69 req/s）。FP8 的引擎调度线程本来就把一个 CPU 核跑满了，对 CPU 侧的额外开销最敏感。**做法**：吞吐只取不开 profiler 的运行；profiler 用过多次的服务，重启后再测。
3. **profile 的时机和文件完整性**：负载还没上来就开始抓，抓到的是空闲；trace 文件还在写就拷走，拿到的是损坏的 gzip。改成等运行中 + 排队请求数达到阈值再开始抓，拷贝前等文件大小稳定并通过 `gzip -t` 校验。
4. **闭环合成负载会误导瓶颈判断**：见第三步。固定并发、同时开始的闭环负载下，瓶颈是 token 预算；泊松到达的真实负载下，瓶颈变成并发槽位。
5. **不是所有 FP8 后端都更快**：MoE 换成 DeepGEMM 后端反而慢了约 30%（TPOT 升到约 26 ms），最后用的是默认的 Triton 后端。
6. **不同节点之间有 2–6% 的差异**：同一配置（FP8、max_num_seqs 128）先后调度到 4 台节点，饱和吞吐分别是 15.9、15.0、15.8、15.0 req/s；256 / 384 档两次测量差约 2%。单点数字按 ±5% 理解，对比实验尽量在同一节点、同一时段内完成，或者多测几台给出区间。

# 待复测清单与后续方向

下表汇总了正文中标 🔁 的地方，以及还没做但值得做的优化。优先级按"对结论的影响"排序。

| 项目 | 现象 / 为什么存疑 | 怎么复测或推进 |
|-|-|-|
| **copy generation 上 FP8 接受率** | 500 条样本上 FP8 少发约 4 个百分点的文案（p = 0.018，校正后约 0.11），多出的是边界样本上的"无相关候选"拒绝。 | 再生成一组 FP8 重跑，看接受率下降能否复现；若复现，评估对业务指标的影响。 |
| **评估器整体偏严** | 所有组（含线上输出）的绝对分都偏低，如选题质量约 0.33。 | 抽样看扣分理由，区分裁判过严和 prompt 普遍问题。 |
| **FP8 max_num_seqs=512 启动失败** | torch.compile 编译 GDN 相关 Triton kernel 时报 `KeyError: 'cubin'`。 | 最小化复现后向上游报告；在新版本 vLLM / PyTorch 上重试。 |
| **AWQ 384 档每步 token 数异常** | profiler 窗口测到的值比 256 档低，与吞吐趋势矛盾。 | 加长采样窗口、多次采样取均值。 |
| **合成负载 32768 档有效算力下降** | 端到端有效算力掉到约 104 TFLOPS，kernel 算力却在上升。 | 用开环泊松压测重跑该档。 |
| **真实负载下的中间预算档** | 只测了 2048、16384、32768，4096 / 8192 未测。 | 补测，确认在 max_num_seqs 放开后预算的最小够用值。 |
| ✅ 效果评估样本量 | 原来只有 100 条。 | 已完成：两个 prompt 各 500 条 + 同配置重跑，见"效果验证"。 |
| ✅ 节点间差异 | 同配置换节点吞吐不同。 | 已完成：4 台节点上测得差异约 2–6%。 |
| ✅ 高并发配置下的速率扫描与 TPOT | 原来只在 max_num_seqs=128 下扫过；384 档中负载 TPOT 曾高 35%。 | 已完成：见图 5b；TPOT 升高未复现。 |
| 后续：**调优 FP8 MoE kernel 配置** | vLLM 没有 H20 上这个专家形状（E=256, N=512）的 FP8 调优配置，用的是 Triton 默认参数。 | 用 vLLM 自带的 `benchmark_moe.py --tune` 生成配置并加载，预计直接提升 MoE kernel 效率。 |
| 后续：**prefill 步用上 CUDA graph** | 当前只有 ≤256 token 的步走 CUDA graph，prefill 步全部 eager 发射。 | 评估扩大捕获范围或分段 graph 的收益和显存代价。 |
| 后续：**前缀对齐 1,056** | 共享前缀只命中 2,112 / 2,570 token。 | 调整系统提示词长度到块边界，预计命中率从 66% 提高。 |

# 小结

- **先找"白算"的计算**：长共享前缀 + prefix cache 是最便宜的 2.5 倍。混合架构要注意块大小和 align 模式带来的整块命中限制。
- **量化的收益要看瓶颈在哪**：FP8 让 kernel 快了 1.7 倍，但当每步计算量太小时，瓶颈会转移到 CPU 发射，端到端收益被吃掉。
- **盯住"每步 token 数"这一个指标**：它决定 GPU 是否吃饱。token 预算和 max_num_seqs 都只是手段，哪个在限制它就调哪个，而且要在真实形态的负载下判断。
- **用 roofline 知道还剩多少**：最终达到上限的 54%，GPU 已不空闲，剩下的空间在 kernel 效率上。
- **效果验证要带噪声参照**：带随机采样的任务，同一配置跑两次也会有差异。用"同配置重跑"作参照、逐条配对比较，才能分清是真的变差还是随机波动；这次它帮我们排除了一个看似显著的"安全分下降"，也找出了 FP8 在copy generation 上少发约 4% 文案这个真实的待确认点。
