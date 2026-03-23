---
date: 2026-03-16
title: PyTorch 基础操作（GPT 实现中的使用）
description: Let's Build GPT 视频中用到的所有 PyTorch API 详解，含示例与可视化
tags: [pytorch, tensors, nn.embedding, softmax, attention, masked-fill, torch.tril]
---

# PyTorch 基础操作

> 本笔记覆盖 [Let's Build GPT](./index) 视频中出现的所有关键 PyTorch API。

---

## 张量操作

### `@`：矩阵乘法

`@` 是 `torch.matmul` 的运算符语法，支持批次维度的**广播矩阵乘法**。

```python
# 二维矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B            # (3, 5)

# 批次矩阵乘法（最后两个维度做乘法，前面广播）
Q = torch.randn(2, 8, 64)   # (B, T, head_size)
K = torch.randn(2, 8, 64)   # (B, T, head_size)
scores = Q @ K.transpose(-2, -1)   # (B, T, T)  ← attention scores
```

**在 GPT 中的使用**：
```python
# 计算 attention weights
wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

# 用 attention weights 对 value 加权求和
out = wei @ v                                        # (B, T, head_size)
```

📖 [torch.matmul 文档](https://pytorch.org/docs/stable/generated/torch.matmul.html)

---

### `torch.tril`：下三角矩阵

生成下三角矩阵（对角线及以下为 1，以上为 0）。用于构造**因果掩码**——确保每个位置只能看到自己和之前的 token。

```python
T = 5
mask = torch.tril(torch.ones(T, T))
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**在 GPT 中的使用**：
```python
# 注册为 buffer（不参与训练，但跟随模型移动到 GPU）
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

# 使用时只取前 T 行/列（实际序列长度）
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

📖 [torch.tril 文档](https://pytorch.org/docs/stable/generated/torch.tril.html)

---

### `tensor.masked_fill`：按掩码填充

将满足条件（mask 为 `True`）的位置替换为指定值。

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

**在 GPT 中的使用**：将未来位置的 attention score 填为 `-inf`，使 softmax 后概率为 0：

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)  # -inf → 0.0（数学上等价）
```

📖 [Tensor.masked_fill 文档](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)

---

### `tensor.view` / `tensor.reshape`：重塑张量

改变张量的形状，不改变数据（`view` 要求内存连续）。

```python
x = torch.randn(4, 8, 65)  # (B, T, C)

# GPT 中：在计算 cross_entropy 前将时间步和批次展平
logits  = logits.view(B*T, C)   # (32, 65)
targets = targets.view(B*T)      # (32,)
```

📖 [Tensor.view 文档](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)

---

### `tensor.transpose`：转置指定维度

```python
k = torch.randn(2, 8, 64)  # (B, T, head_size)
k_T = k.transpose(-2, -1)  # (B, head_size, T)  ← 交换最后两个维度
```

**在 GPT 中**：计算 Q·Kᵀ 时需要转置 K：
```python
wei = q @ k.transpose(-2, -1)  # (B,T,hs) @ (B,hs,T) → (B,T,T)
```

---

### `torch.cat`：拼接张量

沿指定维度拼接多个张量（维度数量不变，该维度大小相加）。

```python
# Multi-Head Attention：将多个 head 的输出拼接
heads_out = [h(x) for h in self.heads]  # 每个: (B, T, head_size)
out = torch.cat(heads_out, dim=-1)       # (B, T, n_embd)

# generate 函数中：将新生成的 token 追加到序列
idx = torch.cat((idx, idx_next), dim=1)  # (B, T) → (B, T+1)
```

📖 [torch.cat 文档](https://pytorch.org/docs/stable/generated/torch.cat.html)

---

### `torch.stack`：堆叠张量

将多个张量沿**新维度**堆叠（维度数量+1）：

```python
# get_batch 中：将多个样本堆叠为批次
x = torch.stack([data[i:i+block_size] for i in ix])  # (batch_size, block_size)
```

`torch.cat` vs `torch.stack`：
- `cat`：在已有维度上拼接，形状 `(3,4)` + `(3,4)` → `(6,4)` 或 `(3,8)`
- `stack`：创建新维度，形状 `(3,4)` + `(3,4)` → `(2,3,4)`

---

### `torch.arange`：创建序列张量

```python
T = 8
pos = torch.arange(T)  # tensor([0, 1, 2, 3, 4, 5, 6, 7])
```

**在 GPT 中**：生成位置索引，传入 position embedding：
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

## Neural Network 模块

### `nn.Embedding`：嵌入查找表

本质是一个可训练的**查找表**（lookup table）：给定整数索引，返回对应的向量。

```python
# vocab_size 行，每行是长度 n_embd 的向量
embedding = nn.Embedding(vocab_size, n_embd)

# 前向传播：整数索引 → 向量
idx = torch.tensor([0, 5, 2])   # 3 个 token
out = embedding(idx)             # (3, n_embd)
```

**GPT 中使用了两个 Embedding**：
```python
# Token embedding：字符 ID → 语义向量
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

# Position embedding：位置 0~T → 位置向量
self.position_embedding_table = nn.Embedding(block_size, n_embd)

# 使用时直接相加
x = tok_emb + pos_emb   # (B, T, n_embd)
```

<HtmlVisualization
  src="/machine-learning/build-gpt-karpathy/visualizations/embedding-lookup.html"
  height="440px"
  title="nn.Embedding 查找过程可视化"
/>

📖 [nn.Embedding 文档](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

---

### `nn.Linear`：线性变换层

实现 $y = xW^T + b$，是 MLP 和 Attention 的基础组件。

```python
linear = nn.Linear(in_features=384, out_features=64, bias=False)
# weight 形状：(64, 384)

x = torch.randn(4, 8, 384)   # (B, T, n_embd)
y = linear(x)                 # (B, T, 64)
```

**GPT 中的使用**：
```python
# Attention 的 Q/K/V 投影（无 bias）
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

# FFN 的两层
nn.Linear(n_embd, 4 * n_embd)
nn.Linear(4 * n_embd, n_embd)

# 输出头（vocab 预测）
self.lm_head = nn.Linear(n_embd, vocab_size)
```

📖 [nn.Linear 文档](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

---

### `nn.LayerNorm`：层归一化

对每个样本的特征维度做归一化（均值为 0，方差为 1），加速训练、稳定梯度。

```python
ln = nn.LayerNorm(n_embd)

x = torch.randn(4, 8, 384)
y = ln(x)   # 对最后一个维度(384)归一化，形状不变: (4, 8, 384)
```

**GPT 中使用 Pre-LayerNorm**（每个子层之前）：
```python
x = x + self.sa(self.ln1(x))    # LayerNorm → Attention → 残差
x = x + self.ffwd(self.ln2(x))  # LayerNorm → FFN → 残差
```

::: info BatchNorm vs LayerNorm
- **BatchNorm**：跨样本归一化（依赖 batch 大小）
- **LayerNorm**：跨特征归一化（每个样本独立），更适合变长序列
:::

📖 [nn.LayerNorm 文档](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

---

### `nn.Dropout`：随机丢弃

训练时随机将部分神经元输出置 0（概率为 `p`），防止过拟合。推理时自动关闭。

```python
dropout = nn.Dropout(p=0.2)

x = torch.randn(4, 8, 384)
y = dropout(x)   # 训练时 20% 的值变为 0，剩余值缩放 1/(1-p)
```

**GPT 中的使用**：
```python
self.dropout = nn.Dropout(dropout)
# 用在 attention weights 后、projection 后、FFN 后
```

📖 [nn.Dropout 文档](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

---

### `nn.ModuleList`：模块列表

将多个子模块注册为 `nn.Module`，使 PyTorch 能正确追踪参数。

```python
# 不能用普通 Python list！
# self.heads = [Head(hs) for _ in range(n)]  ← 参数不会被追踪

# 必须用 ModuleList
self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

# 迭代调用
outputs = [h(x) for h in self.heads]
```

📖 [nn.ModuleList 文档](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

---

### `register_buffer`：注册非参数张量

将张量注册为 buffer：**不参与梯度计算**，但会随模型一起移动到 GPU（`.to(device)`）：

```python
# 在 __init__ 中注册
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

# 使用时通过 self.tril 访问（在 GPU 上）
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

典型场景：固定的掩码矩阵、位置编码等不需要训练的常量。

---

## Functional 操作

### `F.softmax`：Softmax 归一化

将一组数值转换为概率分布（所有值 ≥ 0，和为 1）。

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=-1)
# tensor([0.6590, 0.2424, 0.0986])  — 和为 1

# 处理 -inf（masked attention）
logits_masked = torch.tensor([2.0, -float('inf'), -float('inf')])
F.softmax(logits_masked, dim=-1)
# tensor([1., 0., 0.])  ← 完美地将 -inf 变为 0
```

**在 GPT 中的使用**：
```python
# attention weights 归一化
wei = F.softmax(wei, dim=-1)   # (B, T, T)，每行求和为 1

# generate 时将 logits 转为概率
probs = F.softmax(logits, dim=-1)
```

📖 [F.softmax 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)

---

### `F.cross_entropy`：交叉熵损失

语言模型的标准训练损失，等价于 `log_softmax + NLLLoss`：

$$\mathcal{L} = -\log P(\text{correct token})$$

```python
# logits: (N, C) — N 个样本，C 个类别
# targets: (N,) — 每个样本的正确类别索引
loss = F.cross_entropy(logits, targets)
```

**在 GPT 中**：
```python
B, T, C = logits.shape
loss = F.cross_entropy(
    logits.view(B*T, C),  # (B*T, vocab_size)
    targets.view(B*T)      # (B*T,)
)
```

📖 [F.cross_entropy 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

---

## 采样

### `torch.multinomial`：按概率采样

从概率分布中采样索引，不是取 argmax，而是**按概率随机采样**（保证生成多样性）：

```python
probs = torch.tensor([0.1, 0.6, 0.2, 0.1])

# num_samples=1: 采一个样本（有 60% 概率采到 index=1）
idx = torch.multinomial(probs, num_samples=1)

# replacement=False（默认）: 不放回采样
# replacement=True: 放回采样（可采 num_samples > len(probs) 个）
```

**在 GPT generate 中**：
```python
probs = F.softmax(logits, dim=-1)      # (B, vocab_size)
idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
```

> `argmax`（贪婪）vs `multinomial`（采样）：贪婪解码每次都生成相同文本，而采样有随机性，生成更多样的文本。

📖 [torch.multinomial 文档](https://pytorch.org/docs/stable/generated/torch.multinomial.html)

---

## 训练技巧

### `@torch.no_grad()`

禁用梯度计算，评估时节省内存和加速计算：

```python
@torch.no_grad()
def estimate_loss():
    model.eval()    # 切换到推理模式（关闭 Dropout）
    # ...
    model.train()   # 切换回训练模式
```

### `optimizer.zero_grad(set_to_none=True)`

比 `zero_grad()` 更快：将梯度设为 `None` 而不是 0，节省内存。

### `torch.optim.AdamW`

带权重衰减的 Adam 优化器，GPT 训练的标准选择：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

---

## 快速查阅表

| 操作 | 形状变化示例 | 用途 |
|------|-------------|------|
| `A @ B` | `(B,T,hs) @ (B,hs,T)` → `(B,T,T)` | Attention score 计算 |
| `torch.tril(ones(T,T))` | `(T,T)` 下三角矩阵 | 构造因果掩码 |
| `x.masked_fill(mask, -inf)` | 形状不变 | 遮蔽未来位置 |
| `F.softmax(x, dim=-1)` | 形状不变 | 转为概率分布 |
| `nn.Embedding(V, d)(idx)` | `(B,T)` → `(B,T,d)` | Token/位置嵌入 |
| `nn.Linear(d_in, d_out)(x)` | `(...,d_in)` → `(...,d_out)` | Q/K/V 投影，FFN |
| `nn.LayerNorm(d)(x)` | 形状不变 | 归一化特征维度 |
| `torch.multinomial(probs, 1)` | `(B,V)` → `(B,1)` | 随机采样下一个 token |
| `torch.cat([...], dim=-1)` | `n×(B,T,hs)` → `(B,T,n_embd)` | 拼接多头输出 |
| `x.view(B*T, C)` | `(B,T,C)` → `(B*T,C)` | 展平批次用于 loss |
