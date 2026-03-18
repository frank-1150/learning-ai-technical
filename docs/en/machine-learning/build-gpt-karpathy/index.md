---
title: Let's Build GPT from Scratch (Karpathy)
description: Andrej Karpathy walks through building a GPT from scratch, covering the full path from Bigram to a complete Transformer
tags: [gpt, transformer, karpathy, language-model, self-attention, pytorch]
---

# Let's Build GPT from Scratch

## Video Resources

| Resource | Link |
|----------|------|
| 📺 YouTube | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 🧪 Google Colab | [colab.research.google.com](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) |
| 💻 GitHub Repo | [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) |

---

## What We're Building

- **Task**: Character-level language model
- **Training data**: Shakespeare's complete works (~1MB)
- **Goal**: Given preceding characters, predict the next character
- **Final scale**: ~**10M parameter** GPT model

> Karpathy's core insight: **GPT is essentially a Transformer decoder**, built up step by step from the simplest Bigram model.

---

## Build Path Overview

```
Character tokenization
      ↓
Bigram model (baseline)
      ↓
Math trick: averaging historical context
      ↓
Self-Attention (single head)
      ↓
Scaled Dot-Product Attention
      ↓
Multi-Head Attention
      ↓
Feed-Forward Network
      ↓
Transformer Block (residual + LayerNorm + Dropout)
      ↓
Complete GPT (6-layer stack)
```

---

## Step 1: Character Tokenization

Shakespeare's text contains **65 distinct characters** (including upper/lowercase, punctuation):

```python
chars = sorted(list(set(text)))   # all unique characters
vocab_size = len(chars)           # 65

stoi = { ch:i for i,ch in enumerate(chars) }  # char → int
itos = { i:ch for i,ch in enumerate(chars) }  # int → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

Unlike GPT-2/3 which use a **BPE** tokenizer (~50k tokens), this uses the simplest character-level encoding.

---

## Step 2: Bigram Baseline Model

The simplest language model: predict the next character using only the **current character**:

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
            logits = logits[:, -1, :]              # take last timestep only
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

**Problem**: Bigram completely ignores historical context! Prediction at position `t` has no dependence on `t-2, t-3, ...`.

---

## Step 3: Math Trick — Weighted Average of History

**Goal**: Let each token "see" all previous tokens.

**Simplest implementation**: average all past tokens (bag of words):

```python
# Naive version (loops)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # (t+1, C)
        xbow[b, t] = xprev.mean(0)  # mean
```

**Matrix multiplication version**:

```python
wei = torch.tril(torch.ones(T, T))   # lower triangular (mask)
wei = wei / wei.sum(dim=1, keepdim=True)  # normalize
xbow = wei @ x                       # (T,T) @ (B,T,C) → (B,T,C)
```

::: tip Key Insight
`torch.tril` creates a lower triangular matrix, ensuring position `t` can only see information from `0...t` (**causality**). This is the essence of **masked attention** in GPT!
:::

---

## Step 4: Self-Attention (Single Head)

Upgrading from "average" to "**weighted average**" — each token decides which history to attend to via Query/Key/Value:

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
        k = self.key(x)    # (B, T, head_size) — "what I am"
        q = self.query(x)  # (B, T, head_size) — "what I'm looking for"

        # Attention scores (affinities)
        wei = q @ k.transpose(-2, -1)          # (B, T, T)
        wei = wei * k.shape[-1] ** -0.5        # scale (prevent softmax saturation)

        # Causal mask: can't see the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)           # (B, T, T), each row sums to 1

        v = self.value(x)  # (B, T, head_size) — "what I can provide"
        out = wei @ v      # (B, T, head_size)
        return out
```

**Formula**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/attention-mask.html"
  height="520px"
  title="Causal Attention Mask Visualization"
/>

---

## Step 5: Multi-Head Attention

Run multiple Attention Heads in parallel, letting the model **attend from different perspectives**:

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

- `n_head = 6`, `head_size = n_embd // n_head = 64`
- After concatenation, project back to `n_embd` via a linear layer

---

## Step 6: Feed-Forward Network (FFN)

Attention handles **communication between** tokens; FFN handles **computation within** each token:

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand 4×
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # compress back
            nn.Dropout(dropout),
        )
```

> "**Attention is communication, FFN is computation.**" — Karpathy

---

## Step 7: Transformer Block

Combine Attention + FFN with **residual connections** and **Pre-LayerNorm**:

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # residual + Self-Attention
        x = x + self.ffwd(self.ln2(x))  # residual + FFN
        return x
```

::: warning Note: Pre-LN vs Post-LN
This uses **Pre-LayerNorm** (normalize first, then attention), different from the original Transformer paper's Post-LN, and is more stable during training.
:::

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/transformer-block.html"
  height="480px"
  title="Transformer Block Structure"
/>

---

## Step 8: Complete GPT Model

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        # Token embedding + position embedding
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 6 Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        self.ln_f   = nn.LayerNorm(n_embd)          # final LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size) # output head

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)                    # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T))     # (T, n_embd)
        x = tok_emb + pos_emb                                        # add!
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B,T,vocab_size)
        return logits, loss
```

---

## Final Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_embd` | 384 | embedding dimension |
| `n_head` | 6 | number of attention heads |
| `n_layer` | 6 | number of Transformer layers |
| `block_size` | 256 | context window length |
| `batch_size` | 64 | batch size |
| `dropout` | 0.2 | dropout rate |
| `learning_rate` | 3e-4 | AdamW learning rate |
| **Total parameters** | **~10.7M** | |

---

## Key Takeaways

1. **Causal masking**: `torch.tril` + `masked_fill(-inf)` + `softmax` = only look at the past, the core mechanism
2. **Scaled attention**: divide by `√d_k` to prevent vanishing gradients from large dot products causing softmax saturation
3. **Positional encoding**: token embedding + position embedding added directly (learned, not sinusoidal)
4. **Residual connections**: `x = x + sublayer(x)` allows gradients to flow directly, key to training deep networks
5. **Pre-LayerNorm**: normalize before each sublayer, more stable than the original Transformer

## Related Notes

- [PyTorch Basics](./pytorch-basics) — All PyTorch APIs covered in this video, explained in detail
