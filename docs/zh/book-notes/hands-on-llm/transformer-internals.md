---
date: 2026-04-17
title: "Transformer 架构内部：前向传播、KV 缓存与现代优化"
description: "打开 Decoder-only Transformer 的黑盒：一个 Token 从输入到输出经历了什么？为什么 KV Cache 能把推理速度提升数十倍？MQA/GQA/RoPE 又解决了什么问题？"
tags: [Transformer, Attention, KV-Cache, RoPE, LLM 推理]
---

# Transformer 架构内部：前向传播、KV 缓存与现代优化

> 本文对应原书 **第 3 章 Looking Inside Large Language Models**，覆盖：Decoder-only 前向传播、采样策略、KV 缓存、Transformer Block 组成、MQA/GQA 注意力优化、RoPE 位置编码、架构改进方向。

## 开篇问题：一次前向传播到底发生了什么？

你输入一句 prompt："The capital of France is"，模型返回一个 token "Paris"。看起来很简单的一步，背后要经过：

1. 分词器把字符串切成 token id 序列
2. 每个 id 被查表成一个 3072 维的向量
3. 这串向量流经 32 层 Transformer Block
4. 最后一层的最后一个位置输出一个 3072 维向量
5. LM Head 把它投影成 32064 维（词表大小）的 logits
6. Softmax + 采样策略选出一个 token id
7. 分词器把 id 反解回字符串 "Paris"

本章要做的事，就是把第 3 步"流经 32 层 Transformer Block"这个黑盒打开，同时说明为什么现代推理引擎能把这一过程加速数十倍——答案是 **KV Cache** 加上一组现代架构优化（GQA / RoPE / FlashAttention / RMSNorm / SwiGLU）。

## 一、Decoder-only Transformer 全景图

书中用 Phi-3-mini-4k-instruct 作为解剖对象。打印它的结构，可以看到三段式：

```text
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(...)
        (mlp): Phi3MLP(...)
        (input_layernorm): Phi3RMSNorm()
        (post_attention_layernorm): Phi3RMSNorm()
      )
    )
    (norm): Phi3RMSNorm()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
```

三段的角色分工极其清晰：

| 组件 | 作用 | 可训参数占比 |
|------|------|-------------|
| **Tokenizer + Embedding** | 字符串 ↔ 整数 id ↔ 向量 | ~3%（词表 × 维度） |
| **Stack of Transformer Blocks** | 所有"理解"与"推理"发生的地方，32 层串联 | ~95% |
| **LM Head** | 把最后一层输出投影回词表空间，打分 | ~3%（常与 Embedding 共享权重） |

![Transformer LLM 结构](/book-notes/hands-on-llm/images/transformer-llm-structure.png)

注意："LM Head"并不是什么神秘模块，它就是一个 `Linear(3072, 32064)`。之所以叫 Head，是因为同样的 Transformer 主干可以接不同的 Head——接分类头做分类、接 token 分类头做 NER、接 LM 头做自回归生成。

## 二、前向传播分解：从 id 到 logits

### 2.1 Token Embedding 查表

分词器维护一张 vocab 表（这里 32064 行），每行对应一个 3072 维向量。整个查表过程是一次 `Embedding` 层的 gather 操作，本质是"把整数当作行索引去取向量"——几乎零计算量。

### 2.2 位置信息注入

纯 Embedding 只反映"是哪个 token"，并不反映"在哪一位"。早期 Transformer 会在 Embedding 上加一个绝对位置向量；现代模型（Llama 2/3、Phi-3、Mistral）则改成把位置信息注入到注意力里的 Query / Key——这就是后面要讲的 **RoPE**。

### 2.3 每一层 Transformer Block

每一层做的事情抽象成一句话：**"以一个 token 为中心，把上下文中相关的信息揉进它的表示里"**。Block 内部有两个关键子模块：

- **Self-Attention**：从前面所有 token 中收集信息
- **Feed-Forward Network (FFN / MLP)**：在当前 token 的表示里做非线性变换，可以粗略理解为"记忆 + 插值"

每个子模块外面再包一圈 **Residual 残差连接 + LayerNorm**（现代模型用 RMSNorm 并改成 pre-norm）。

![Transformer Block](/book-notes/hands-on-llm/images/transformer-block.png)

### 2.4 LM Head 与 logits

32 层走完后，每个位置都有一个 3072 维向量。对**最后一个位置**做 `Linear(3072, 32064)`，得到 32064 维的 logits。这就是"词表里每个 token 作为下一个 token 的未归一化打分"。

::: tip 为什么只取最后一个位置？
并行计算时所有位置都算了 logits，但只有"当前位置要预测下一个 token"这个意图真正需要用到。训练时我们用所有位置的 logits 算 cross-entropy（每个位置预测下一个），**推理生成时只用最后一个**。这是一个容易让初学者困惑的点。
:::

## 三、采样策略：从概率分布中"挑一个"

LM Head 的输出是一串 logits，经过 softmax 变成概率分布。如何从这个分布里挑一个 token，决定了生成的"性格"：

| 策略 | 做法 | 特点 |
|------|------|------|
| **Greedy (argmax)** | 永远选概率最高的 | 确定性，无随机；易陷入重复；`temperature=0` 等价于它 |
| **Temperature 采样** | logits 除以 T 再 softmax，T<1 更尖锐、T>1 更平坦 | T→0 退化为 greedy；T→∞ 退化为均匀采样 |
| **Top-K** | 只在概率最高的 K 个 token 中采样 | 简单暴力，K 难调 |
| **Top-P (Nucleus)** | 从累计概率达到 p 的最小 token 集合中采样 | 动态调整候选数，主流选择 |

实践里常见组合是 `temperature=0.7, top_p=0.9`——让模型有创造性但不脱轨。作者的原话是：

> 贪心解码 ≠ 最佳生成。适度的随机性反而让输出更自然，因为语言本身就不是"最高概率唯一"的。

## 四、并行计算与上下文限制

### 4.1 Prefill 阶段是并行的

书里反复强调一个直觉：**Transformer 的每个 token 都走一条独立的计算流**。把 prompt 里的 6 个 token 塞进模型，模型会给每个位置产生一条"stream"——6 条流并行穿过 32 层。

这 6 条流彼此之间只在 **Self-Attention 里才会交互**：第 6 条流在算注意力时，会把前面 5 条流的 Key / Value 拿来做 "relevance scoring + combining"。其他步骤（Embedding、FFN、LayerNorm）在位置维度上完全独立——所以可以塞进一个大矩阵乘法里一次性算完，这叫 **Prefill**（预填充）。

### 4.2 Decode 阶段是串行的

但一旦开始"生成新 token"，就没办法并行了——你要拿第 N+1 个 token 去预测第 N+2 个，必须等 N+1 算完才行。这一步叫 **Decode**。

所以 LLM 推理的两个阶段在计算特性上完全不同：

| 阶段 | 计算量 | 并行性 | 瓶颈 |
|------|--------|--------|------|
| **Prefill**（处理 prompt） | O(n²) 注意力 | 高度并行 | 算力（compute-bound） |
| **Decode**（一次吐一个 token） | O(n) 注意力（有 KV Cache 时） | 不可并行 | 显存带宽（memory-bound） |

### 4.3 上下文长度为什么受限？

注意力矩阵是 `n × n` 的（n 是序列长度），内存 / 计算都随 n² 增长。同时 KV Cache 也随 n 线性增长（乘上所有层、所有 head）。这是"上下文长度越长越贵"的直接原因。

## 五、KV Cache：推理加速的核心

::: warning 这是本章最重要的一节
KV Cache 不是"锦上添花"，它是**让 LLM 推理在工程上可行**的前提。书里给的对比数据：Colab T4 上生成 100 个 token，**开 KV Cache 用 4.5 秒，关掉用 21.8 秒**——4.8× 的差距。实际生产环境上，这个比例往往更夸张。
:::

### 5.1 为什么能缓存

生成第 N+1 个 token 时，模型要对整段序列（1..N）重新跑一次前向，但其中每一层的 Self-Attention 需要的三样东西：

- **Query**：只需要当前这一个位置的（位置 N）
- **Key**：1..N 所有位置的
- **Value**：1..N 所有位置的

关键洞察是：**对于前面 1..(N-1) 位置的 Key / Value，上一步生成时就已经算过，而且值不会改变**（因为它们是 `W_k · x` 和 `W_v · x`，x 不变 → K、V 不变）。所以把它们缓存起来，下一步只算位置 N 这一个新 token 的 Q、K、V，然后把新的 K、V 拼到缓存上即可。

![KV Cache](/book-notes/hands-on-llm/images/transformer-kvcache.png)

### 5.2 空间换时间

- **无 KV Cache**：生成第 N 个 token 时，要重新算 N 个位置的 K、V，单步复杂度 O(N)；总复杂度 O(N²)
- **有 KV Cache**：生成第 N 个 token 时，只算 1 个位置的 K、V，单步复杂度 O(1)；总复杂度 O(N)

单步从 O(N) 降到 O(1)，这就是 20× 加速的来源。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/kv-cache-flow.html" height="600px" title="KV Cache 工作原理" />

### 5.3 显存占用公式

KV Cache 要额外占多少显存？

```
KV Cache 大小 = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes × batch_size
              ↑
            (K 和 V 两份)
```

以 **Llama-2-7B** 为例：`n_layers=32, n_heads=32, head_dim=128, fp16=2 bytes`，单序列每个 token 占的 KV Cache：

```
2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB / token
```

4K 上下文就是 2 GB；32K 上下文 16 GB——这就是为什么"上下文越长显存越炸"。**这也正是后面 MQA / GQA 要解决的问题**：把 `n_heads` 从 K/V 的公式里拿掉一大半。

## 六、Transformer Block 拆解

一个现代（2024 年代）的 Transformer Block 长这样：

![2024 Transformer Block](/book-notes/hands-on-llm/images/transformer-block-2024.png)

数据流走向（pre-norm 版本）：

```text
x → RMSNorm → Self-Attention(GQA + RoPE) → + residual
  → RMSNorm → FFN(SwiGLU)                → + residual
  → 下一层
```

相比原始 Transformer 论文有四处关键改动：

1. **Pre-norm 代替 post-norm**：LayerNorm 放在子模块**前面**而不是后面，训练更稳
2. **RMSNorm 代替 LayerNorm**：去掉 mean 中心化，只除以 RMS，计算更快
3. **SwiGLU 代替 ReLU**：门控非线性，效果更好
4. **RoPE 代替绝对位置编码**：在 attention 内部注入位置信息

### 6.1 Self-Attention 的两步

Self-Attention 做两件事：

**第一步，Relevance Scoring**：把当前位置的 Query 向量和之前每个位置的 Key 向量做点积，得到一个"相关性打分"向量，再过 softmax 归一化。

![Attention 打分](/book-notes/hands-on-llm/images/transformer-attention-score.png)

**第二步，Combining Information**：用这些打分作为权重，对之前每个位置的 Value 向量做加权求和——得到一个新的向量，就是注意力层对当前位置的输出。

数学形式就是大家背过的公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

直觉解释："我（Q）在问一个问题，扫过全文所有位置（K）看谁跟我这个问题最相关，然后把他们的"内容"（V）按相关性加权汇总回来"。

### 6.2 Multi-Head：让一个 token 同时"关注多个方向"

单个 Attention Head 只能学到一种注意力模式（比如"指代消解"）。把 Q/K/V 在维度上切成 h 份（`d_model = h × d_head`），每一份独立做一次注意力，最后拼回来——这就是 Multi-Head Attention。

Llama-2-7B 是 32 头、head_dim=128、d_model=4096。Phi-3-mini 是 32 头、head_dim=96、d_model=3072。

### 6.3 Feed-Forward (FFN / MLP)

书中原话把 FFN 的作用概括得很好：

> FFN 是模型**存储知识、做插值**的地方。输入 "The Shawshank" 最终输出 "Redemption"，这个记忆就压在 FFN 的权重里。

FFN 的结构是"升维 → 激活 → 降维"——先把 d_model 投到 4 × d_model（或 8/3 × d_model，用于 SwiGLU），过激活函数，再投回来。**FFN 承担了 Transformer 大部分的参数量**（约 2/3）。

## 七、现代注意力优化：MHA → MQA → GQA

### 7.1 三种变体的核心差异

所有优化都围绕同一个目标：**减少 K / V 头的数量，从而减少 KV Cache**。

![MHA vs GQA vs MQA](/book-notes/hands-on-llm/images/transformer-mha-gqa-mqa.png)

| 变体 | Q 头数 | K/V 头数 | KV Cache 大小（相对 MHA） | 代表模型 |
|------|--------|----------|--------------------------|---------|
| **MHA (Multi-Head)** | h | h | 1× | GPT-2/3、原始 Transformer |
| **MQA (Multi-Query)** | h | 1 | 1/h | PaLM、Falcon |
| **GQA (Grouped-Query)** | h | g（1 < g < h） | g/h | **Llama 2 / 3, Mistral, Phi-3** |

### 7.2 为什么 GQA 胜出

- **MQA 太激进**：所有头共享一组 K/V，信息瓶颈过窄，模型质量明显下滑
- **MHA 太浪费**：KV Cache 全员在线，显存跑得比算力快
- **GQA 是折中**：Llama-2-70B 用 g=8（64 个 Q 头 + 8 个 KV 头），KV Cache 砍到 1/8，质量几乎无损

Llama-3-70B 的 KV Cache 具体是多少？

```
Llama-3-70B: n_layers=80, n_q_heads=64, n_kv_heads=8, head_dim=128
单 token KV Cache = 2 × 80 × 8 × 128 × 2 bytes = 327,680 bytes ≈ 0.31 MB

对比如果是 MHA（64 KV heads）：
2 × 80 × 64 × 128 × 2 bytes = 2.6 MB / token（8 倍）
```

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/attention-variants.html" height="550px" title="注意力变体对比 MHA/MQA/GQA" />

### 7.3 Sparse Attention：另一条路线

GQA 减的是"头"，Sparse Attention 减的是"看到的 token 范围"。

- **Local / Sliding-Window Attention**（Longformer、Mistral）：每个 token 只关注最近的 w 个 token，注意力从 O(n²) 降到 O(n·w)
- **Strided / Fixed Sparse Attention**（Sparse Transformer、GPT-3）：按固定模式跳着关注
- **Full + Sparse 混合**：GPT-3 交替用全注意力和稀疏注意力的层，兼顾质量与效率

### 7.4 FlashAttention：算法层面的 I/O 优化

前面几种是"减少计算或缓存量"；FlashAttention 走的是另一条路——**完全不改数学定义，只优化 GPU 上的内存访问**。

GPU 上的内存分层：`HBM（大、慢） ↔ SRAM（小、快）`。原始 attention 实现会把 n×n 的注意力矩阵写回 HBM 再读出来——这是瓶颈。FlashAttention 用 **tiling（分块）+ recomputation（反向重算）** 让 attention 直接在 SRAM 里完成，避免了 n×n 中间矩阵的 HBM 往返。

结果：**训练 / 推理都能快 2-4×，同时精度完全不变**。现代推理引擎（vLLM、SGLang、TGI）默认都开 FlashAttention-2 或更新版本。

## 八、RoPE：旋转位置编码

### 8.1 为什么不用绝对位置编码

原始 Transformer 在 Embedding 上加绝对位置向量（position i 加一个固定向量 pe_i）。这套方案有几个致命问题：

1. **外推性差**：训练时只见过 position 0..2047，推理时喂到 position 4000 直接崩
2. **相对位置不友好**：模型实际关心的是"A 和 B 相隔多远"，而不是"A 在第几个、B 在第几个"
3. **Packing 不友好**：训练时为了效率会把多份短文档拼进一个 context（中间加 `<sep>`），绝对位置会给 Document 2 错误的"我在全局第 500 位"的信号

### 8.2 RoPE 的思路：旋转

RoPE 的核心直觉：**把位置信息表达成对向量的"旋转角度"**。第 i 个位置的 Q（或 K）被旋转 `i × θ` 弧度（θ 是频率参数，不同维度对用不同频率）。

数学上，对 2 维子向量 (x, y) 施加位置 i 的旋转：

$$
\begin{pmatrix} x' \\ y' \end{pmatrix} =
\begin{pmatrix} \cos(i\theta) & -\sin(i\theta) \\ \sin(i\theta) & \cos(i\theta) \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}
$$

然后它的神奇性质是：**两个旋转过的向量做点积，结果只跟"两者旋转角度之差"有关**，也就是"相对位置"。这就在内积层面天然注入了相对位置信息。

![RoPE 注入](/book-notes/hands-on-llm/images/transformer-rope.png)

### 8.3 RoPE 的实际好处

- **相对位置**：attention score 天然只依赖相对距离，更符合语言直觉
- **外推性强**：配合 NTK / YaRN 之类的技术可以把训练时的 4K context 延伸到 32K、128K
- **在 attention 内部注入，不改 Embedding**：对 packing 友好，对 KV Cache 友好
- **无需额外参数**：旋转矩阵由正弦函数计算得到，不可训练

主流开源模型（Llama 2/3、Mistral、Qwen、Phi-3、DeepSeek）几乎都用 RoPE。

## 九、其他架构改进速览

| 改进 | 替代了谁 | 带来的好处 |
|------|---------|-----------|
| **RMSNorm** | LayerNorm | 去掉均值中心化，计算更快；训练更稳 |
| **SwiGLU** | ReLU / GELU | 门控结构，同等参数下效果更好（Gated Linear Unit 家族） |
| **Pre-norm** | Post-norm | 深层网络梯度更稳，可以叠到 100+ 层 |
| **Parallel Layers** | 串行 Attention + FFN | Attention 和 FFN 并行计算（GPT-J 风格），提速但略损质量 |
| **MoE (Mixture of Experts)** | 单个 FFN | 每个 token 只激活一部分 FFN 专家；Mixtral、DeepSeek-V2/V3 核心机制 |

## 十、小结

| 关键词 | 一句话解释 |
|--------|-----------|
| **自回归生成** | 每次输出一个 token，把它拼回 prompt，再预测下一个 |
| **前向传播三段** | Tokenizer + Embedding → N × Transformer Block → LM Head |
| **Prefill vs Decode** | 处理 prompt 时算力密集、可并行；生成时显存带宽密集、必须串行 |
| **KV Cache** | 缓存前面所有 token 的 K/V，把单步推理从 O(n) 降到 O(1) |
| **MHA / MQA / GQA** | 减少 K/V 头数的三种选择，Llama 2/3 用 GQA |
| **FlashAttention** | 不改数学、只改 I/O，让 attention 直接在 SRAM 完成 |
| **RoPE** | 用旋转在 attention 内部注入相对位置信息，外推性好 |
| **Pre-norm / RMSNorm / SwiGLU** | 2024 年代 Transformer Block 的标配三件套 |

打开 Transformer 的黑盒之后，下一章将走出"模型原理"世界，进入"怎么用 LLM 解决具体问题"——从文本分类与聚类开始。
