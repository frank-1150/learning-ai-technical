---
title: Let's Build GPT from Scratch (Karpathy)
description: Andrej Karpathy 手把手从零构建 GPT，覆盖从 Bigram 到完整 Transformer 的完整路径
tags: [gpt, transformer, karpathy, language-model, self-attention, pytorch]
---

# Let's Build GPT from Scratch

## 视频资源

| 资源 | 链接 |
|------|------|
| 📺 YouTube 视频 | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 🧪 Google Colab | [colab.research.google.com](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) |
| 💻 GitHub Repo | [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) |

---

## 我们在构建什么

- **任务**：字符级语言模型（character-level language model）
- **训练数据**：Shakespeare 全集文本（~1MB）
- **目标**：给定前面的字符，预测下一个字符
- **最终规模**：约 **10M 参数**的 GPT 模型

> Karpathy 的核心思想：**GPT 本质上就是 Transformer decoder**，从最简单的 Bigram 出发，一步步自然推导出整个架构。

---

## 构建路径总览

```
字符 tokenization
      ↓
Bigram 模型（基线）
      ↓
数学 trick：平均历史上下文
      ↓
Self-Attention（单头）
      ↓
Scaled Dot-Product Attention
      ↓
Multi-Head Attention
      ↓
Feed-Forward Network
      ↓
Transformer Block（残差 + LayerNorm + Dropout）
      ↓
完整 GPT（6层堆叠）
```

---

## Step 1：字符 Tokenization

Shakespeare 文本共有 **65 个不同字符**（包括大小写、标点）：

```python
chars = sorted(list(set(text)))   # 所有唯一字符
vocab_size = len(chars)           # 65

stoi = { ch:i for i,ch in enumerate(chars) }  # char → int
itos = { i:ch for i,ch in enumerate(chars) }  # int → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

与 GPT-2/3 使用 **BPE** tokenizer（~50k tokens）不同，这里用最简单的字符级编码。

---

## Step 2：Bigram 基线模型

最简单的语言模型：只看**当前一个字符**预测下一个：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)   # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # 只取最后一个时间步
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

**问题**：Bigram 完全没有利用历史上下文！位置 `t` 的预测与 `t-2, t-3, ...` 无关。

---

## Step 3：数学 Trick —— 加权平均历史

**目标**：让每个 token 能"看到"它之前的所有 token。

**最简单实现**：对过去所有 token 求平均（bag of words）：

```python
# 低效版（循环）
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # (t+1, C)
        xbow[b, t] = xprev.mean(0)  # 均值
```

**矩阵乘法版本**：

```python
wei = torch.tril(torch.ones(T, T))   # 下三角矩阵（掩码）
wei = wei / wei.sum(dim=1, keepdim=True)  # 归一化
xbow = wei @ x                       # (T,T) @ (B,T,C) → (B,T,C)
```

::: tip 关键洞见
`torch.tril` 创建下三角矩阵，确保位置 `t` 只能看到 `0...t` 的信息（**因果性 / causal**）。这就是 GPT 中 **masked attention** 的本质！
:::

---

## Step 4：Self-Attention（单头）

从"平均"升级到"**加权平均**"——每个 token 通过 Query/Key/Value 决定关注哪些历史：

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size) — "我是什么"
        q = self.query(x)  # (B, T, head_size) — "我在找什么"

        # 计算 attention scores（亲和力）
        wei = q @ k.transpose(-2, -1)          # (B, T, T)
        wei = wei * k.shape[-1] ** -0.5        # 缩放（防止 softmax 饱和）

        # 因果掩码：不能看未来
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)           # (B, T, T)，每行求和为 1

        v = self.value(x)  # (B, T, head_size) — "我能提供什么"
        out = wei @ v      # (B, T, head_size)
        return out
```

**公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/attention-mask.html"
  height="520px"
  title="因果 Attention 掩码可视化"
/>

---

## Step 5：Multi-Head Attention

并行运行多个 Attention Head，让模型**从不同角度**关注信息：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

- `n_head = 6`，`head_size = n_embd // n_head = 64`
- 拼接后通过线性层投影回 `n_embd`

---

## Step 6：Feed-Forward Network（FFN）

Attention 做的是 token **之间**的通信，FFN 做的是每个 token **内部**的计算：

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展 4 倍
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 压缩回来
            nn.Dropout(dropout),
        )
```

> "**Attention is communication, FFN is computation.**" — Karpathy

---

## Step 7：Transformer Block

将 Attention + FFN 组合，加入**残差连接**和 **Pre-LayerNorm**：

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 残差 + Self-Attention
        x = x + self.ffwd(self.ln2(x))  # 残差 + FFN
        return x
```

::: warning 注意：Pre-LN vs Post-LN
这里用的是 **Pre-LayerNorm**（先归一化再注意力），与原始 Transformer 论文的 Post-LN 不同，训练更稳定。
:::

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/transformer-block.html"
  height="480px"
  title="Transformer Block 结构图"
/>

---

## Step 8：完整 GPT 模型

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        # Token embedding + 位置 embedding
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 6 个 Transformer Block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        self.ln_f   = nn.LayerNorm(n_embd)          # 最后的 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size) # 输出头

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)                    # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T))     # (T, n_embd)
        x = tok_emb + pos_emb                                        # 相加！
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B,T,vocab_size)
        return logits, loss
```

---

## 最终超参数

| 参数 | 值 | 含义 |
|------|----|------|
| `n_embd` | 384 | embedding 维度 |
| `n_head` | 6 | 注意力头数 |
| `n_layer` | 6 | Transformer 层数 |
| `block_size` | 256 | 上下文窗口长度 |
| `batch_size` | 64 | 批大小 |
| `dropout` | 0.2 | dropout 率 |
| `learning_rate` | 3e-4 | AdamW 学习率 |
| **总参数量** | **~10.7M** | |

---

## 关键知识点总结

1. **Causal masking**：`torch.tril` + `masked_fill(-inf)` + `softmax` = 只看过去，核心机制
2. **Scaled attention**：除以 `√d_k` 防止点积过大导致 softmax 梯度消失
3. **位置编码**：token embedding + position embedding 直接相加（learned，非 sinusoidal）
4. **残差连接**：`x = x + sublayer(x)` 让梯度直接流过，训练深层网络的关键
5. **Pre-LayerNorm**：每个子层前归一化，比原始 Transformer 更稳定

## 相关笔记

- [PyTorch 基础操作](./pytorch-basics) — 本视频中所有 PyTorch API 详解
