---
date: 2026-04-17
title: "微调生成模型：SFT、LoRA/QLoRA 与 DPO 对齐"
description: "把 LLM 从通用基座调成你想要的样子，需要走完三步：预训练、监督微调、偏好对齐。本文重点讲清后两步——LoRA 如何用 0.1% 参数完成微调、QLoRA 如何在消费级 GPU 上跑 70B，以及 DPO 如何简化 RLHF。"
tags: [SFT, LoRA, QLoRA, DPO, RLHF, PEFT, Fine-tuning]
---

# 微调生成模型：SFT、LoRA/QLoRA 与 DPO 对齐

> 本文对应原书 **第 12 章 Fine-Tuning Generation Models**，覆盖：LLM 训练三阶段、全量 vs 参数高效微调、LoRA/QLoRA 原理与实战、生成模型评估方法、RLHF 与 DPO 对齐。

---

## 1. LLM 训练的三阶段：从预测下一个 token 到"像人一样回答"

一个真正能用的 LLM（比如 ChatGPT、Claude、Llama-3-Instruct），背后几乎都走过同一条三段式训练路径：

![LLM 训练三阶段（原书 Figure 12-3）](/book-notes/hands-on-llm/images/ft-three-stages.png)

### 1.1 Pretraining：语言建模，学会"说人话"

第一步是 **语言建模（language modeling）**：拿海量语料（互联网、书籍、代码），让模型通过"预测下一个 token"的自监督任务，学到词汇、句法、世界知识。这个阶段产出的叫 **base model** 或 **foundation model**。

- 目标：最小化下一个 token 的交叉熵
- 数据：TB 级、无标签
- 算力：数千张 H100，训练数周到数月

但 base model 有个问题：它只会"接着往下写"，不会"回答你"。你问它 `What is 1+1?`，它可能会"接着"生成 `2. What is 1+1+1? 3. What is 1+1+1+1? ...`——因为训练语料里这种"问题列表"太常见了。

### 1.2 Supervised Fine-Tuning (SFT)：学会听指令

第二步用**高质量的"指令-回答"对**做监督微调（Instruction Tuning）。结构上仍然是预测下一个 token，但只对 assistant 的回答部分计算 loss。

做完 SFT 之后，模型就会把 `What is 1+1?` 当成一个需要回答的指令，输出 `The answer to 1+1 is 2!`。这一步产出的叫 **instruction-tuned / chat model**。

### 1.3 Preference Tuning：对齐人类偏好

第三步是 **偏好调优**：用人类对"哪个回答更好"的判断，进一步打磨模型的风格、安全性、事实性。这一步的经典方法是 **RLHF（Reinforcement Learning from Human Feedback）**，新一代方法是 **DPO（Direct Preference Optimization）**。

三阶段对比：

| 阶段 | 目标 | 数据量 | 数据形态 | 算力 |
|------|------|--------|---------|------|
| Pretraining | 学会语言 | 数 T token | 无标签文本 | 超大（千卡级） |
| SFT | 学会按指令输出 | 数万到数百万 | `(instruction, response)` | 中等（单机到多机） |
| Preference Tuning | 对齐偏好 | 数千到数十万 | `(prompt, chosen, rejected)` | 较小（单机） |

对一线工程师来说，**预训练几乎不可能自己做**（算力成本几百万美金起），但 **SFT 和 Preference Tuning 在消费级到企业级 GPU 上都能做**——这正是本章的重点。

---

## 2. Full Fine-Tuning：最直接但最昂贵的方案

Full fine-tuning（全量微调）和预训练一个 LLM 几乎一样，区别只是：**用小一些、带标签的数据集**训练全部参数。

![Full Fine-Tuning 流程（原书 Figure 12-6）](/book-notes/hands-on-llm/images/ft-lora.png)

它的问题在于显存。我们算一下：训练一个 7B 参数的模型需要多少显存？

| 开销 | 公式 | 7B 模型（FP16） |
|------|------|---------|
| 模型权重 | N × 2 bytes | 14 GB |
| 梯度 | N × 2 bytes | 14 GB |
| Adam 优化器状态 | N × 8 bytes（m + v，FP32） | 56 GB |
| 激活值 | 取决于 batch/seq len | 10–30 GB |
| **合计** | ≈ N × 16–20 bytes | **90–120 GB** |

换句话说，**7B 模型全量微调要 80GB+ 的 A100**，70B 模型更是只能在多节点集群上跑。消费级 24GB 4090 显然无望。于是，**PEFT（Parameter-Efficient Fine-Tuning）** 成了绝大多数工程实践的首选。

> [!tip]
> 一个粗略估算公式：**全量微调显存 ≈ 参数量 × (16–20) bytes**。7B → ~100GB，13B → ~200GB，70B → ~1TB。这就是为什么没人在自己的 4090 上全量微调 Llama-3。

---

## 3. PEFT 与 LoRA：只训 0.1% 的参数，效果打平

### 3.1 Adapters：PEFT 的原始思路

PEFT 最早的代表是 **Adapters**（Houlsby et al., 2019）：在 Transformer 每个 block 里插入一些小的模块（通常是 down-projection → 非线性 → up-projection），只训练这些 adapter，主干冻结。

论文在 BERT 上证明：**只微调 3.6% 的参数，GLUE 成绩就能达到全量微调 99.6% 的水平**。

![Adapter 在 Transformer block 里的位置（原书 Figure 12-8）](/book-notes/hands-on-llm/images/ft-qlora.png)

Adapter 的优点是可插拔：同一个 base 模型配不同 adapter 就能胜任不同任务。但缺点也明显：**推理时多了两次矩阵运算，延迟变高**。

### 3.2 LoRA：真正的工业级 PEFT

2021 年微软提出的 **LoRA（Low-Rank Adaptation）**，几乎替代了 Adapter 成为事实标准。

**核心观察**：论文 *"Intrinsic dimensionality explains the effectiveness of language model fine-tuning"* 发现，LLM 微调时权重的**更新量 ΔW 实际上是低秩的**——即 ΔW 可以用两个很"瘦"的矩阵乘积来近似。

**数学形式**：假设原权重 `W ∈ R^{d×d}`，LoRA 把更新量分解为：

```
ΔW = B · A
其中 A ∈ R^{r×d}、B ∈ R^{d×r}，r ≪ d
```

前向传播变成 `h = W x + B A x`。训练时 **只更新 A、B 这两个小矩阵，W 冻结**。参数量从 `d²` 降到 `2 × d × r`——如果 `d=4096, r=8`，参数量从 16.7M 降到 66K，**压缩约 250 倍**。

![LoRA vs Full Fine-Tuning（原书 Figure 12-14）](/book-notes/hands-on-llm/images/ft-reward-model.png)

> [!note]
> GPT-3 的一个 weight matrix 是 12288 × 12288 = 151M 参数。用 rank=8 的 LoRA，只需 2 × 12288 × 8 = 197K 参数——约 0.13%。而 GPT-3 有 96 个 transformer block，节省的参数量是天文数字。

**推理时可以无损 merge 回原权重**：`W' = W + BA`，之后推理路径和原模型完全一样，**零额外延迟**——这是 LoRA 相对 Adapter 最大的优势。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/lora-decomposition.html" height="550px" title="LoRA 低秩分解原理（可交互）" />

### 3.3 LoRA 的关键超参

| 超参 | 作用 | 典型值 |
|------|------|--------|
| `r` | 秩，控制压缩程度 | 4、8、16、32、64 |
| `lora_alpha` | 缩放因子，等效学习率放大 `α/r` | 通常取 `2r` |
| `target_modules` | 对哪些层加 LoRA | `q_proj`, `v_proj` 是最常见；激进做法全部都加 |
| `lora_dropout` | LoRA 层的 dropout | 0.0–0.1 |
| `bias` | 是否训练 bias | `none` / `lora_only` / `all` |

**经验法则**：
- **r 越大，表达能力越强，但参数量和显存也线性增加**。从 8 或 16 起步，不够再加。
- **target_modules 建议加全**：Query、Key、Value、Output、MLP 的 up/down/gate。覆盖面广了，r 可以小一些。
- **lora_alpha 约等于 r 的 2 倍**是个广为流传的经验值（含义是等效学习率 α/r=2）。

### 3.4 QLoRA：让 33B 模型塞进 24GB GPU

LoRA 已经很省了，但 base 模型本身还是 FP16——7B 依然要 14GB，33B 要 66GB，消费级 GPU 还是吃不下。

**QLoRA**（Dettmers et al., 2023）的思路：**把冻结的主干模型量化到 4-bit**，LoRA 层仍然用 16-bit 训练。

三项关键创新：

1. **NF4（4-bit NormalFloat）**：传统 INT4 是均匀量化，但神经网络权重近似正态分布——大部分值集中在 0 附近。NF4 把量化点按正态分布分布，**让"常见的值"有更高精度**。

![NF4 按分布感知分桶（原书 Figure 12-18）](/book-notes/hands-on-llm/images/ft-rlhf.png)

2. **Double Quantization**：量化所用的 scale 也量化一次，再省 0.5 bit/参数左右。
3. **Paged Optimizer**：借助 NVIDIA 统一内存，把优化器状态分页到 CPU 内存里，应对显存峰值。

效果：**33B 模型在 24GB 的 4090 上能跑**；65B 可以在单张 A100 80GB 上微调——在 QLoRA 出现之前，这是想都不敢想的。

| 方案 | 7B 模型微调显存 | 说明 |
|------|---------------|------|
| Full Fine-Tuning | ~100 GB | 必须 A100 80GB × 2 |
| LoRA (FP16) | ~20 GB | 一张 3090/4090 勉强 |
| **QLoRA (NF4)** | **~7 GB** | 甚至 T4 / Colab 免费 GPU |

---

## 4. 指令微调实战：TinyLlama + QLoRA

原书用 TinyLlama（1.1B 参数）演示了完整的 QLoRA 流程。下面是精简版代码——完整可以在 Colab 免费 T4 上跑完。

### 4.1 指令数据模板化

不同模型有不同的 chat template。TinyLlama-Chat 用的是：

```
<|user|>
What is 1+1?</s>
<|assistant|>
The answer to 1+1 is 2!</s>
```

关键是用 `<|user|>` 和 `<|assistant|>` 标记角色，`</s>` 作为 EOS。`apply_chat_template` 会自动做这个拼接：

```python
from transformers import AutoTokenizer
from datasets import load_dataset

template_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1BChat-v1.0")

def format_prompt(example):
    chat = example["messages"]
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

dataset = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    .shuffle(seed=42)
    .select(range(3_000))
)
dataset = dataset.map(format_prompt)
```

UltraChat 是 ~20 万条用户与 LLM 的高质量对话，这里取 3000 条做演示。

### 4.2 4-bit 加载 base 模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit 加载
    bnb_4bit_quant_type="nf4",          # NormalFloat 4
    bnb_4bit_compute_dtype="float16",   # 计算时转回 fp16
    bnb_4bit_use_double_quant=True,     # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device_map="auto",
    quantization_config=bnb_config,
)
model.config.use_cache = False
```

加载完成后，1.1B 模型只占 **~1 GB** 显存（原本 16-bit 要 ~4 GB）。

### 4.3 LoRA 配置

```python
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

peft_config = LoraConfig(
    r=64,                  # 秩
    lora_alpha=32,         # 缩放（注意这里用 α<r，是原书的选择）
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "gate_proj", "v_proj", "up_proj",
                    "q_proj", "o_proj", "down_proj"],  # 几乎全覆盖
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
```

`prepare_model_for_kbit_training` 会自动处理一些量化训练的细节：layer-norm 保持 FP32、启用 gradient checkpointing 等。

### 4.4 训练配置与训练

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,       # 等效 batch size = 8
    optim="paged_adamw_32bit",           # QLoRA 推荐的分页优化器
    learning_rate=2e-4,                  # LoRA 常用较大学习率
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    peft_config=peft_config,
)
trainer.train()
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")
```

`SFTTrainer`（来自 `trl` 库）是 `Trainer` 的一个薄包装，专门为 SFT 场景准备。在 Colab T4 上这个训练大约跑 1 小时。

### 4.5 合并权重

训练出来的只是 LoRA adapter（几 MB）。要部署，需要把 adapter merge 回 base model：

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
)
merged_model = model.merge_and_unload()  # 合并 & 丢弃 LoRA 结构
```

注意这里要用 **FP16 加载 base（不是 4-bit）**，再 merge——量化下 merge 精度会损失。Merge 完就得到一个标准的 HF 模型，可以 `push_to_hub` 或用 vLLM 部署。

---

## 5. 评估生成模型：一个没有标准答案的问题

生成模型评估的核心困难是：**同一个问题可能有多种"对"的回答**，单一指标很难判断好坏。原书给出的方法可以分为四类：

### 5.1 词级指标（Word-Level Metrics）

| 指标 | 原理 | 典型场景 |
|------|------|---------|
| **Perplexity** | 衡量模型对一段文本的"困惑度"——预测下一个 token 的概率越高越低 | 语言建模 |
| **BLEU** | n-gram 重叠率，有长度惩罚 | 机器翻译 |
| **ROUGE** | 基于 recall 的 n-gram 重叠 | 文本摘要 |
| **BERTScore** | 用 BERT 做语义级相似度 | 开放生成 |

这些指标的问题：**它们只看表面词汇重叠，不关心语义、一致性、流畅度、正确性**。一段生成"这是一个句子"可能 BLEU/PPL 都很好，但信息量为零。

### 5.2 能力基准（Benchmarks）

| Benchmark | 测什么 | 备注 |
|-----------|-------|------|
| **MMLU** | 57 个学科的多选题（数学、历史、医学……） | 综合能力 |
| **GSM8K** | 小学数学应用题 | 推理能力 |
| **HellaSwag** | 常识推理（四选一） | 世界知识 |
| **TruthfulQA** | 事实性 / 拒绝捏造 | 幻觉检测 |
| **HumanEval** | 164 个编程题（Python） | 代码能力 |
| **GLUE** | 经典 NLU 任务合集 | 语言理解 |

benchmarks 优点是**标准化、可复现**；缺点是：**容易被数据泄漏污染、模型可能过拟合到题目格式、无法覆盖所有下游场景**。

### 5.3 榜单（Leaderboards）

**Open LLM Leaderboard**（HuggingFace 维护）是目前最主流的开源 LLM 排行榜，集合 HellaSwag、MMLU、TruthfulQA、GSM8K 等 6 个 benchmark，综合打分。

**Chatbot Arena**（LMSYS）则是人工评估的代表：随机展示两个匿名 LLM 的回答，让用户选更好的，用 Elo rating 计算排名。截至原书写作，已收集 80 万 + 人工投票。

### 5.4 LLM-as-Judge：自动化评估新范式

用一个更强的 LLM（比如 GPT-4）作评委，对两个模型的输出做 pairwise 比较。代表工具：

- **MT-Bench**：80 个多轮对话题，GPT-4 打分
- **AlpacaEval**：805 条指令，GPT-4 比较胜率

优点是**便宜、快、开放题也能评**；缺点是**评委 LLM 自己有偏好（如偏爱更长的回答、偏爱自家模型风格）**。

### 5.5 人工评估：金标准但昂贵

人类评估永远是 ground truth，但贵、慢、难规模化。实践中的典型做法：

- **冒烟测试**（10–50 条自己关心的 prompt，自己逐条过）
- **A/B 测试**（线上真实用户，用 thumbs up/down 或满意度收集数据）

> [!warning]
> **Goodhart's Law**: *When a measure becomes a target, it ceases to be a good measure.* 盯着某个 benchmark 优化会让模型在该 benchmark 上漂亮但在别处退化。**一定要结合多个指标 + 你自己的 use case 评估**。

---

## 6. Preference Tuning：为什么只做 SFT 还不够？

做完 SFT 后，模型已经会"按指令回答"，但仍然存在几个常见问题：

- **幻觉**：编造不存在的事实
- **啰嗦 or 过于简短**：格式不稳定
- **有害输出**：在极端 prompt 下可能输出偏见、仇恨言论
- **风格不统一**：不同用户期望的语气差异大

偏好调优（Preference Tuning）的思路：**收集一批"pair 偏好数据"（同一个 prompt 下哪个回答更好），用它进一步调模型**。这一步不需要"告诉模型标准答案是什么"，只需要"告诉模型哪个更好"——这让数据标注成本大幅下降。

### 6.1 RLHF 的三步流程

经典 RLHF（OpenAI 在 InstructGPT 和 ChatGPT 里用的）分三步：

![RLHF 流程：SFT → Reward Model → PPO（原书 Figure 12-30）](/book-notes/hands-on-llm/images/ft-dpo.png)

1. **SFT**：基础对话能力（上一节已经完成）
2. **训练 Reward Model (RM)**：
   - 输入：`(prompt, response)`
   - 输出：单一 scalar 分数
   - 架构：把 SFT 模型的 LM head 换成一个 scalar 回归 head
   - 训练数据：`(prompt, chosen, rejected)` pairs，目标是让 `score(chosen) > score(rejected)`
3. **用 PPO 强化学习优化 LLM**：用 RM 的分数当 reward，用 Proximal Policy Optimization 更新 LLM，同时加一个 KL 散度惩罚项，防止模型偏离 SFT 模型太远。

### 6.2 RLHF 的痛点

- **需要训练/维护 2 个模型**（RM + LLM）：显存翻倍
- **PPO 对超参极度敏感**：KL 系数、clip range、value loss 权重都不好调
- **训练不稳定**：reward hacking（模型学会"骗"RM 拿高分但实际更差）
- **工程复杂**：需要 rollout、advantage 估计、on-policy 采样……

这些问题让 RLHF 一度只有大厂才做得起。

---

## 7. DPO：直接偏好优化，干掉 Reward Model

2023 年 Stanford 的 Rafailov 等人提出 **DPO（Direct Preference Optimization）**，论文副标题极具挑衅：*"Your Language Model is Secretly a Reward Model"*。

**核心洞察**：在 RLHF 的数学推导里，最优策略和 reward model 有一个闭式对应关系。这意味着**可以跳过显式训练 RM，直接用偏好数据对 LLM 做一个"分类式"优化**。

### 7.1 DPO 的直觉

![DPO 的核心结构：reference model + trainable model（原书 Figure 12-32）](/book-notes/hands-on-llm/images/ft-reward-model.png)

DPO 有两个模型：
- **Reference model（冻结）**：一般就是 SFT 之后的模型
- **Trainable model**：初始化与 reference 相同，准备微调

训练目标（直觉版）：
- **对 chosen 回答**：让 trainable model 的概率 ↑，相对 reference 的对数比增大
- **对 rejected 回答**：让 trainable model 的概率 ↓，相对 reference 的对数比减小

具体 loss（sigmoid 版二分类）：

```
L_DPO = -log σ( β · [log π(chosen) - log π_ref(chosen)]
              - β · [log π(rejected) - log π_ref(rejected)] )
```

其中 β 控制"偏离 reference 多远"（类似 RLHF 的 KL 系数，典型值 0.1–0.5）。

### 7.2 DPO vs RLHF 对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要的模型 | SFT + RM + LLM（+ value net） | SFT (frozen ref) + trainable |
| 训练阶段 | 2 步（RM → PPO） | 1 步 |
| 数据需求 | pair 偏好数据 | pair 偏好数据 |
| 稳定性 | 差，对超参敏感 | 好，类似监督学习 |
| 实现复杂度 | 高（rollout、advantage） | 低（加载 2 个模型 + 一个 loss） |
| 最终质量 | 略高（精调后） | 相当，某些任务更好 |

**一句话总结**：**DPO 把 RLHF 从"强化学习"降维成了"监督学习"**——只需要一个类 cross-entropy 的损失函数，其他的基础设施几乎可以复用 SFT。

### 7.3 DPO 实战：基于 trl 的 DPOTrainer

数据格式（来自 argilla/distilabel-intel-orca-dpo-pairs）：

```python
{
    "prompt": "<|system|>\n...\n</s>\n<|user|>\nExplain quantum entanglement.</s>\n<|assistant|>\n",
    "chosen": "Quantum entanglement is a phenomenon where...</s>\n",
    "rejected": "Entanglement is when particles are tangled up.</s>\n",
}
```

训练代码：

```python
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM

# 加载之前 SFT 后 merge 好的模型（仍 4-bit 量化）
model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    device_map="auto",
    quantization_config=bnb_config,
)
merged_model = model.merge_and_unload()

# 继续套 LoRA 做 DPO
peft_config = LoraConfig(
    r=64, lora_alpha=32, lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "gate_proj", "v_proj", "up_proj",
                    "q_proj", "o_proj", "down_proj"],
)

training_arguments = DPOConfig(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,         # DPO 用更小的学习率
    lr_scheduler_type="cosine",
    max_steps=200,
    warmup_ratio=0.1,           # 前 10% warmup
    fp16=True,
    gradient_checkpointing=True,
)

dpo_trainer = DPOTrainer(
    merged_model,
    args=training_arguments,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,                   # 偏离 reference 的惩罚强度
    max_prompt_length=512,
    max_length=512,
)
dpo_trainer.train()
dpo_trainer.model.save_pretrained("TinyLlama-1.1B-dpo-qlora")
```

注意几个关键细节：
- **不需要显式传 reference model**：`trl` 会自动把 LoRA 关掉时的 base 当 reference（省一份显存）
- **学习率比 SFT 小 20 倍**（2e-4 → 1e-5）：DPO 对学习率敏感
- **max_steps=200** 就够：DPO 数据通常少、epoch 也不用多，过多反而劣化

### 7.4 DPO 之后：ORPO、KTO、IPO 等

DPO 开启了"免 RL 对齐"的一大家族：

- **ORPO**（2024）：把 SFT 和 preference tuning 合成一步，完全不需要 reference model，训练成本再减半
- **KTO**（Kahneman-Tversky Optimization）：只需要"这个回答是好 / 坏"的二分类信号，不需要 pair
- **IPO**（Identity Preference Optimization）：修正 DPO 在高置信偏好上的过拟合问题
- **SimPO**：进一步简化，连 reference model 都不要

对工业界来说，DPO 已经成为事实标准，是否切换到 ORPO / KTO 取决于数据形态。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/alignment-comparison.html" height="620px" title="五种微调/对齐方法多维对比" />

---

## 8. 小结：选哪种微调方法？

给不同场景的建议：

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 学习 LLM 微调原理 | QLoRA SFT | Colab 免费 T4 就能跑通 |
| 垂直领域知识注入（医疗、法律） | QLoRA SFT | 全量成本太高、LoRA 够用 |
| 改变模型风格/格式 | QLoRA SFT | 小数据（几千条）就见效 |
| 对齐偏好（安全、有用性） | **DPO（或 ORPO）** | 比 RLHF 简单稳定 |
| 追求 SOTA 质量 | SFT + RLHF / SFT + DPO + RLHF | 大厂级投入 |
| 资源极度紧张（边缘设备） | QLoRA + 后续量化部署 | 端到端都压到 4-bit |

整本书第 12 章的核心信息其实就三句话：

1. **LoRA 让你用 0.1% 的参数跑赢全量微调**——这是过去 5 年 NLP 工程最重要的一个 hack。
2. **QLoRA 让消费级 GPU 也能微调 33B+ 模型**——这让开源社区的 LLM 定制化真正普及。
3. **DPO 把 RLHF 从"强化学习黑魔法"降维成了"加强版 cross-entropy"**——这让"对齐"对普通团队也变得可行。

合在一起，就是当下开源 LLM 工程实践的主线：**QLoRA SFT → DPO → Merge → Deploy**。这条路径对绝大多数团队来说，已经足够把一个开源 base model 调成真正能用的产品。

---

## 参考资料

- Houlsby et al., *"Parameter-Efficient Transfer Learning for NLP"*, ICML 2019（Adapters）
- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, arXiv:2106.09685
- Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs"*, arXiv:2305.14314
- Aghajanyan et al., *"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"*, arXiv:2012.13255
- Rafailov et al., *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"*, arXiv:2305.18290
- Hong et al., *"ORPO: Monolithic Preference Optimization without Reference Model"*, arXiv:2403.07691
- Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347
- Zheng et al., *"Judging LLM-as-a-judge with MT-Bench and Chatbot Arena"*, NeurIPS 2024
