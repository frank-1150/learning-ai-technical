---
title: PyTorch Basics (Used in GPT Implementation)
description: All PyTorch APIs used in the Let's Build GPT video, with examples and visualizations
tags: [pytorch, tensors, nn.embedding, softmax, attention, masked-fill, torch.tril]
---

# PyTorch Basics

> This note covers all key PyTorch APIs that appear in the [Let's Build GPT](./index) video.

---

## Tensor Operations

### `@`: Matrix Multiplication

`@` is the operator syntax for `torch.matmul`, supporting **broadcasting batch matrix multiplication**.

```python
# 2D matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B            # (3, 5)

# Batch matrix multiplication (last two dims multiply, rest broadcast)
Q = torch.randn(2, 8, 64)   # (B, T, head_size)
K = torch.randn(2, 8, 64)   # (B, T, head_size)
scores = Q @ K.transpose(-2, -1)   # (B, T, T)  ← attention scores
```

**Used in GPT**:
```python
# Compute attention weights
wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

# Weighted sum of values using attention weights
out = wei @ v                                        # (B, T, head_size)
```

📖 [torch.matmul docs](https://pytorch.org/docs/stable/generated/torch.matmul.html)

---

### `torch.tril`: Lower Triangular Matrix

Generates a lower triangular matrix (1s on and below the diagonal, 0s above). Used to construct a **causal mask** — ensuring each position can only see itself and preceding tokens.

```python
T = 5
mask = torch.tril(torch.ones(T, T))
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**Used in GPT**:
```python
# Register as buffer (not trainable, but moves to GPU with the model)
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

# Use only the first T rows/cols (actual sequence length)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

📖 [torch.tril docs](https://pytorch.org/docs/stable/generated/torch.tril.html)

---

### `tensor.masked_fill`: Mask-based Filling

Replaces values where the mask is `True` with a specified value.

```python
x = torch.tensor([[1., 2., 3.],
                   [4., 5., 6.],
                   [7., 8., 9.]])

mask = torch.tensor([[False, True,  True],
                     [False, False, True],
                     [False, False, False]])

x.masked_fill(mask, float('-inf'))
# tensor([[  1., -inf, -inf],
#         [  4.,   5., -inf],
#         [  7.,   8.,   9.]])
```

**Used in GPT**: Fill future positions' attention scores with `-inf`, so softmax yields 0 probability:

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)  # -inf → 0.0 (mathematically equivalent)
```

📖 [Tensor.masked_fill docs](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)

---

### `tensor.view` / `tensor.reshape`: Reshape Tensor

Changes tensor shape without changing data (`view` requires contiguous memory).

```python
x = torch.randn(4, 8, 65)  # (B, T, C)

# In GPT: flatten timesteps and batch before computing cross_entropy
logits  = logits.view(B*T, C)   # (32, 65)
targets = targets.view(B*T)      # (32,)
```

📖 [Tensor.view docs](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)

---

### `tensor.transpose`: Transpose Specific Dimensions

```python
k = torch.randn(2, 8, 64)  # (B, T, head_size)
k_T = k.transpose(-2, -1)  # (B, head_size, T)  ← swap last two dimensions
```

**In GPT**: transpose K when computing Q·Kᵀ:
```python
wei = q @ k.transpose(-2, -1)  # (B,T,hs) @ (B,hs,T) → (B,T,T)
```

---

### `torch.cat`: Concatenate Tensors

Concatenates multiple tensors along a specified dimension (number of dimensions stays the same, that dimension's size adds up).

```python
# Multi-Head Attention: concatenate outputs from all heads
heads_out = [h(x) for h in self.heads]  # each: (B, T, head_size)
out = torch.cat(heads_out, dim=-1)       # (B, T, n_embd)

# In generate: append newly generated token to sequence
idx = torch.cat((idx, idx_next), dim=1)  # (B, T) → (B, T+1)
```

📖 [torch.cat docs](https://pytorch.org/docs/stable/generated/torch.cat.html)

---

### `torch.stack`: Stack Tensors

Stacks multiple tensors along a **new dimension** (number of dimensions +1):

```python
# In get_batch: stack multiple samples into a batch
x = torch.stack([data[i:i+block_size] for i in ix])  # (batch_size, block_size)
```

`torch.cat` vs `torch.stack`:
- `cat`: concatenates on an existing dimension, shape `(3,4)` + `(3,4)` → `(6,4)` or `(3,8)`
- `stack`: creates a new dimension, shape `(3,4)` + `(3,4)` → `(2,3,4)`

---

### `torch.arange`: Create Sequence Tensor

```python
T = 8
pos = torch.arange(T)  # tensor([0, 1, 2, 3, 4, 5, 6, 7])
```

**In GPT**: generate position indices for position embedding:
```python
pos_emb = self.position_embedding_table(torch.arange(T, device=device))
```

---

### `tensor.shape` / `B, T, C = x.shape`

```python
x = torch.randn(4, 8, 384)
B, T, C = x.shape   # B=4 (batch), T=8 (time/seq_len), C=384 (channels/n_embd)
```

---

## Neural Network Modules

### `nn.Embedding`: Embedding Lookup Table

Essentially a trainable **lookup table**: given an integer index, returns the corresponding vector.

```python
# vocab_size rows, each row is a vector of length n_embd
embedding = nn.Embedding(vocab_size, n_embd)

# Forward pass: integer indices → vectors
idx = torch.tensor([0, 5, 2])   # 3 tokens
out = embedding(idx)             # (3, n_embd)
```

**GPT uses two Embeddings**:
```python
# Token embedding: character ID → semantic vector
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

# Position embedding: position 0~T → position vector
self.position_embedding_table = nn.Embedding(block_size, n_embd)

# Usage: simply add them
x = tok_emb + pos_emb   # (B, T, n_embd)
```

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/embedding-lookup.html"
  height="440px"
  title="nn.Embedding Lookup Visualization"
/>

📖 [nn.Embedding docs](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

---

### `nn.Linear`: Linear Transformation Layer

Implements $y = xW^T + b$, the fundamental building block of MLPs and Attention.

```python
linear = nn.Linear(in_features=384, out_features=64, bias=False)
# weight shape: (64, 384)

x = torch.randn(4, 8, 384)   # (B, T, n_embd)
y = linear(x)                 # (B, T, 64)
```

**Used in GPT**:
```python
# Q/K/V projections in Attention (no bias)
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

# FFN layers
nn.Linear(n_embd, 4 * n_embd)
nn.Linear(4 * n_embd, n_embd)

# Output head (vocab prediction)
self.lm_head = nn.Linear(n_embd, vocab_size)
```

📖 [nn.Linear docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

---

### `nn.LayerNorm`: Layer Normalization

Normalizes across the feature dimension per sample (mean=0, variance=1), speeding up training and stabilizing gradients.

```python
ln = nn.LayerNorm(n_embd)

x = torch.randn(4, 8, 384)
y = ln(x)   # normalizes over last dim (384), shape unchanged: (4, 8, 384)
```

**GPT uses Pre-LayerNorm** (before each sublayer):
```python
x = x + self.sa(self.ln1(x))    # LayerNorm → Attention → residual
x = x + self.ffwd(self.ln2(x))  # LayerNorm → FFN → residual
```

::: info BatchNorm vs LayerNorm
- **BatchNorm**: normalizes across samples (depends on batch size)
- **LayerNorm**: normalizes across features (each sample independent), better suited for variable-length sequences
:::

📖 [nn.LayerNorm docs](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

---

### `nn.Dropout`: Random Dropout

During training, randomly sets some neuron outputs to 0 (with probability `p`), preventing overfitting. Automatically disabled during inference.

```python
dropout = nn.Dropout(p=0.2)

x = torch.randn(4, 8, 384)
y = dropout(x)   # 20% of values become 0 during training; remaining values scaled 1/(1-p)
```

**Used in GPT**:
```python
self.dropout = nn.Dropout(dropout)
# Applied after attention weights, after projection, after FFN
```

📖 [nn.Dropout docs](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

---

### `nn.ModuleList`: Module List

Registers multiple submodules as `nn.Module` so PyTorch can properly track their parameters.

```python
# Can't use a plain Python list!
# self.heads = [Head(hs) for _ in range(n)]  ← parameters won't be tracked

# Must use ModuleList
self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

# Iterate and call
outputs = [h(x) for h in self.heads]
```

📖 [nn.ModuleList docs](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

---

### `register_buffer`: Register Non-Parameter Tensors

Registers a tensor as a buffer: **does not participate in gradient computation**, but moves to GPU with the model (`.to(device)`):

```python
# Register in __init__
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

# Access via self.tril (already on GPU)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

Typical use cases: fixed mask matrices, positional encodings, and other constants that don't need training.

---

## Functional Operations

### `F.softmax`: Softmax Normalization

Converts a set of values into a probability distribution (all values ≥ 0, sum to 1).

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=-1)
# tensor([0.6590, 0.2424, 0.0986])  — sums to 1

# Handling -inf (masked attention)
logits_masked = torch.tensor([2.0, -float('inf'), -float('inf')])
F.softmax(logits_masked, dim=-1)
# tensor([1., 0., 0.])  ← perfectly converts -inf to 0
```

**Used in GPT**:
```python
# Normalize attention weights
wei = F.softmax(wei, dim=-1)   # (B, T, T), each row sums to 1

# Convert logits to probabilities in generate
probs = F.softmax(logits, dim=-1)
```

📖 [F.softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)

---

### `F.cross_entropy`: Cross-Entropy Loss

The standard training loss for language models, equivalent to `log_softmax + NLLLoss`:

$$\mathcal{L} = -\log P(\text{correct token})$$

```python
# logits: (N, C) — N samples, C classes
# targets: (N,) — correct class index for each sample
loss = F.cross_entropy(logits, targets)
```

**In GPT**:
```python
B, T, C = logits.shape
loss = F.cross_entropy(
    logits.view(B*T, C),  # (B*T, vocab_size)
    targets.view(B*T)      # (B*T,)
)
```

📖 [F.cross_entropy docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

---

## Sampling

### `torch.multinomial`: Probability-Based Sampling

Samples indices from a probability distribution — not argmax, but **random sampling by probability** (ensures generation diversity):

```python
probs = torch.tensor([0.1, 0.6, 0.2, 0.1])

# num_samples=1: sample one index (60% chance of sampling index=1)
idx = torch.multinomial(probs, num_samples=1)

# replacement=False (default): sampling without replacement
# replacement=True: sampling with replacement (can sample num_samples > len(probs))
```

**In GPT generate**:
```python
probs = F.softmax(logits, dim=-1)      # (B, vocab_size)
idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
```

> `argmax` (greedy) vs `multinomial` (sampling): greedy decoding always generates the same text, while sampling has randomness and produces more diverse text.

📖 [torch.multinomial docs](https://pytorch.org/docs/stable/generated/torch.multinomial.html)

---

## Training Techniques

### `@torch.no_grad()`

Disables gradient computation during evaluation, saving memory and speeding up computation:

```python
@torch.no_grad()
def estimate_loss():
    model.eval()    # switch to inference mode (Dropout disabled)
    # ...
    model.train()   # switch back to training mode
```

### `optimizer.zero_grad(set_to_none=True)`

Faster than `zero_grad()`: sets gradients to `None` instead of 0, saving memory.

### `torch.optim.AdamW`

Adam optimizer with weight decay, the standard choice for GPT training:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

---

## Quick Reference Table

| Operation | Shape Change Example | Purpose |
|-----------|----------------------|---------|
| `A @ B` | `(B,T,hs) @ (B,hs,T)` → `(B,T,T)` | Attention score computation |
| `torch.tril(ones(T,T))` | `(T,T)` lower triangular | Construct causal mask |
| `x.masked_fill(mask, -inf)` | shape unchanged | Mask future positions |
| `F.softmax(x, dim=-1)` | shape unchanged | Convert to probability distribution |
| `nn.Embedding(V, d)(idx)` | `(B,T)` → `(B,T,d)` | Token/position embedding |
| `nn.Linear(d_in, d_out)(x)` | `(...,d_in)` → `(...,d_out)` | Q/K/V projections, FFN |
| `nn.LayerNorm(d)(x)` | shape unchanged | Normalize feature dimension |
| `torch.multinomial(probs, 1)` | `(B,V)` → `(B,1)` | Randomly sample next token |
| `torch.cat([...], dim=-1)` | `n×(B,T,hs)` → `(B,T,n_embd)` | Concatenate multi-head outputs |
| `x.view(B*T, C)` | `(B,T,C)` → `(B*T,C)` | Flatten batch for loss computation |
