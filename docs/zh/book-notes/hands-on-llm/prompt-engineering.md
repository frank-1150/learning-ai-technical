---
date: 2026-04-17
title: "Prompt Engineering 与推理技巧：从 Few-Shot 到 Tree-of-Thought"
description: "同一个模型，为什么换个 prompt 准确率能差出 30 个百分点？系统梳理 Prompt 工程的基本构成、In-Context Learning、CoT/Self-Consistency/ToT 三种推理范式，以及如何让输出可靠。"
tags: [Prompt Engineering, Chain-of-Thought, Few-shot, LLM 应用]
---

# Prompt Engineering 与推理技巧：从 Few-Shot 到 Tree-of-Thought

> 本文对应原书 **第 6 章 Prompt Engineering**，覆盖：文本生成模型的参数控制、Prompt 基本构成、In-Context Learning、CoT/Self-Consistency/Tree-of-Thought 三种推理策略、输出约束。

## 为什么 Prompt Engineering 值得专门学

和 BERT 式分类模型不同，文本生成模型（GPT 一派）最大的"操作面"不是"再训练一个分类头"，而是**通过输入文本（prompt）来改变输出行为**。同一个 Phi-3-mini 或 Llama-3-8B，一个精心设计的 prompt 能把数学应用题的准确率从 17% 推到 78%（书中引用的 CoT 论文数据），也可能因为一句"Let's think step by step"就把零样本 GSM8K 成绩翻倍。

作者用一个很犀利的比喻：

> Prompt engineering is, in a way, an attempt to **reverse engineer what the model has learned and how it responds to certain prompts**.

这一章要解决三件事：

1. **文本生成模型怎么用**：选型、加载、用 `temperature` / `top_p` 控制随机性。
2. **好 prompt 的结构长什么样**：从单行指令到七要素复杂 prompt。
3. **怎么让模型"会思考"**：CoT、Self-Consistency、ToT 三种逐级增强的推理范式，以及如何约束输出格式。

## 文本生成模型：从加载到参数控制

### 选型：先从小模型开始

书里没有一上来就用 70B 的模型，而是选了 **Phi-3-mini（3.8B 参数）**。原因很朴素：

- 单张 8GB VRAM 的 GPU 就能跑；
- 小模型对 prompt 更敏感，prompt engineering 的效果更明显；
- 主要是为了理解概念，不是要在榜单上刷分。

加载代码用 Hugging Face 的标准姿势：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)
```

几个不起眼但重要的参数：

- `return_full_text=False`：只返回新生成的部分，不把 prompt 重复吐回来。
- `max_new_tokens=500`：**必须**显式设，否则模型可能停不下来。
- `do_sample=False`：关闭采样，只取概率最高的 token（贪心解码），结果每次完全一致。

### Chat Template：对话模型的"隐形 prompt 结构"

很多人以为传给模型的 prompt 就是那句 "Create a funny joke about chickens."，其实 `pipeline` 在背后把它套进了一个 **chat template**：

```
<s><|user|>
Create a funny joke about chickens.<|end|>
<|assistant|>
```

`<|user|>`、`<|assistant|>`、`<|end|>` 是训练时用的**特殊 token**，告诉模型"现在轮到 assistant 说话了"、"这里是一段完整的 user 发言结束"。不同模型的 template 不一样（Phi-3 和 Llama-3 就完全不同），用错会让输出质量大幅下降。

::: tip 实践建议
永远用 `tokenizer.apply_chat_template(messages, tokenize=False)` 来生成 prompt，不要自己手拼 `<|user|>` 这种标签——`transformers` 会根据模型配置自动选对模板。
:::

### 采样参数详解：为什么 `temperature=0` 不总是最好

语言模型本质是在每个位置上输出一个**概率分布**（所有 token 的 likelihood）。参数控制"从这个分布里怎么挑下一个 token"。

![采样参数：temperature vs top_p](/book-notes/hands-on-llm/images/prompt-sampling.png)

#### temperature：整体"温度"调节

`temperature` 是 softmax 里的除数，值越高概率分布越**平坦**（低概率 token 的相对权重上升）：

- `temperature = 0`：确定性输出，每次选最高概率的 token。
- `temperature = 0.2`：接近贪心，但允许极少量偏离。
- `temperature = 0.8`：明显的随机性，输出更"有创意"。
- `temperature = 2.0`：接近均匀分布，输出经常变成胡言乱语。

#### top_k：硬截断 top-k 个 token

只在**概率最高的 k 个** token 里采样，其余全部屏蔽。例如 `top_k=100` 就只从前 100 个候选里选。

#### top_p（Nucleus Sampling）：按累计概率截断

按概率从高到低排序，累加直到总和达到 `top_p`，只在这部分"核"（nucleus）里采样：

- `top_p = 0.1`：极保守，只考虑最头部的那几个 token。
- `top_p = 0.9`：标准设置，既保留多样性又过滤垃圾。
- `top_p = 1.0`：考虑全部 token（相当于不过滤）。

**top_k 和 top_p 的差别**：top_k 的候选数固定；top_p 的候选数随分布形状动态变化——分布尖锐时候选少，分布平坦时候选多。大多数现代实践里 `top_p` 比 `top_k` 更常用。

#### max_tokens：安全阀

不设会有两种灾难：

1. 模型一直生成到上下文窗口上限，浪费钱；
2. 某些模型在特定 prompt 下陷入无限循环（比如反复输出 "I don't know. I don't know. ..."）。

#### 典型场景参数建议

| 场景 | `temperature` | `top_p` | 说明 |
|------|---------------|---------|------|
| 头脑风暴 | 高（0.8–1.0） | 高（0.9–1.0） | 多样、允许跳脱 |
| 邮件生成 | 低（0.1–0.3） | 低（0.3–0.5） | 稳定、可预测、保守 |
| 创意写作 | 高（0.8–1.0） | 低（0.3–0.5） | 小池子里挑出奇词，仍然连贯 |
| 翻译 | 低（0.1–0.3） | 高（0.9） | 用词准确但允许表达多样性 |
| 代码生成 | 0–0.2 | 0.95 | 几乎确定，但保留合理变体空间 |

## Prompt 的七大要素

作者把一个"完整"的 prompt 拆成七个可组合模块，不是说每次都要全用，而是当输出不满意时知道"该加什么"：

![复杂 prompt 的七要素组合](/book-notes/hands-on-llm/images/prompt-structure.png)

1. **Persona（身份）**：告诉模型扮演谁。"You are an expert in large language models."
2. **Instruction（指令）**：这次到底要做什么。"Summarize the key findings of the paper provided."
3. **Context（背景）**：为什么要做这件事。"This summary will be used to brief busy researchers."
4. **Format（格式）**：输出长什么样。"Bullet-point summary followed by a concise paragraph."
5. **Audience（受众）**：给谁看。ELI5（"Explain like I'm 5"）还是专业读者。
6. **Tone（语气）**：正式、口语、俏皮。
7. **Data（数据）**：实际的输入内容本身。

对应的代码组织方式：

```python
persona      = "You are an expert in Large Language models. "
instruction  = "Summarize the key findings of the paper provided. "
context      = "Your summary should extract the most crucial points. "
data_format  = "Create a bullet-point summary followed by a concise paragraph. "
audience     = "The summary is for busy researchers needing the latest trends. "
tone         = "The tone should be professional and clear. "
data         = f"Text to summarize: {text}"

query = persona + instruction + context + data_format + audience + tone + data
```

这种拼接式写法的好处是**模块化**——你可以一个一个地打开/关闭某个组件，观察输出变化，从而知道哪个模块对你的用例贡献最大。

### 提高 prompt 质量的通用技巧

作者给出几条独立于场景的工程建议：

- **Specificity（具体性）**：不要写 "Write a product description"，而要写 "Write a product description **in less than two sentences using a formal tone**"。
- **Hallucination 控制**：显式告诉模型 "If you do not know the answer, respond with 'I don't know.'"——这能显著减少胡编。
- **Order（位置）**：把指令放在 prompt 的**开头或末尾**，不要埋在中间。引用的论文是 Nelson Liu 的 *"Lost in the Middle: How Language Models Use Long Contexts"*——LLM 天然存在 primacy（首位效应）和 recency（末位效应），对中间内容的注意力明显更弱。

## In-Context Learning：用例子教模型

### 原理：不是"学"，是"模式匹配"

LLM 在推理时**不会更新参数**。所谓 "few-shot learning"，本质是在 prompt 里放几个示范，让模型在上下文里看到"这种任务长什么样"，然后在同一个 forward pass 里模仿。

![zero-shot / one-shot / few-shot 三种形态](/book-notes/hands-on-llm/images/prompt-fewshot.png)

- **Zero-shot**：只给指令，不给例子。依赖模型从预训练里泛化。
- **One-shot**：给 1 个例子。
- **Few-shot**：给 2 个及以上例子。

书中一个漂亮的例子：让模型用一个**自造的词** "Gigamuru"（虚构的日本乐器）造句：

```python
one_shot_prompt = [
    {
        "role": "user",
        "content": "A 'Gigamuru' is a type of Japanese musical instrument. "
                   "An example of a sentence that uses the word Gigamuru is:"
    },
    {
        "role": "assistant",
        "content": "I have a Gigamuru that my uncle gave me as a gift. "
                   "I love to play it at home."
    },
    {
        "role": "user",
        "content": "To 'screeg' something is to swing a sword at it. "
                   "An example of a sentence that uses the word screeg is:"
    },
]

outputs = pipe(one_shot_prompt)
# -> During the intense duel, the knight skillfully screeged his opponent's shield, ...
```

只一个示范就足以让模型"get"到这个范式。

### 选例子的几条原则

- **多样性**：如果做分类，正负样本都要给，不要 5 个都是 positive。
- **贴近目标任务**：要做医疗摘要，就别放新闻摘要当例子。
- **控制长度**：例子会吃掉上下文窗口；太多样本反而让模型被例子中的噪声误导。
- **格式统一**：每个例子的结构（`Text:` / `Sentiment:` 这种前缀）要一致，否则模型可能模仿"格式的变化"本身。

## Chain Prompting：把复杂任务拆成几步

有些任务单个 prompt 塞不下或者一次想不清楚——比如"给产品写名字、口号、销售话术"。Chain prompting 的思路是：**把一个大问题拆成一条调用链，每一步的输出作为下一步的输入**。

```python
# Step 1: 产品名字
name_msg = [{"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}]
product_desc = pipe(name_msg)[0]["generated_text"]

# Step 2: 拿名字和口号去写 sales pitch
pitch_msg = [{
    "role": "user",
    "content": f"Generate a very short sales pitch for the following product: '{product_desc}'"
}]
sales_pitch = pipe(pitch_msg)[0]["generated_text"]
```

这种做法的好处：

- **每步可独立调参**：名字那步用低 temperature 保证简短稳定；sales pitch 那步调高 temperature 保留表达力。
- **便于调试**：哪一步坏了一眼就看出来。
- **便于接入工具**：某一步可以接检索、计算器、数据库查询。

典型的 chain 类型：

- **Response validation**：让模型再跑一次，检查自己刚才的输出是不是合理。
- **Parallel prompts**：并行让多个模型/多个 prompt 生成候选，最后合并。
- **Writing stories**：先写大纲 → 再写角色设定 → 再写分章节情节。

Chain prompting 是下一章 LangChain 的心智基础。

## 推理范式 I：Chain-of-Thought（CoT）

### 魔法咒语 "Let's think step by step"

Kahneman 在《Thinking, Fast and Slow》里把人的思维分成两套系统：

- **System 1**：快速、直觉、自动（你看到 2+2 立刻反应 4）。
- **System 2**：慢速、审慎、需要努力（你算 17×23 需要一步步来）。

普通的 LLM 生成更接近 System 1——看到 prompt 立刻吐答案。CoT 的本质是**把模型推向 System 2**：先把推理过程写出来，再基于推理过程写答案。

::: tip 为什么 CoT 有效？
每生成一个新 token 都会消耗一次 forward pass 的计算。如果模型**直接**生成最终答案，就只有这一次计算的机会；如果它先生成一长串中间推理 token，每个中间 token 都会被后续 token 当作 context 再次参与计算——相当于给了模型"更多时间思考"。这种"test-time compute scaling"正是 2024 年以来 o1、R1 等推理模型的核心思想。
:::

![CoT 对比：one-shot（错 27）vs chain-of-thought（对 9）](/book-notes/hands-on-llm/images/prompt-cot.png)

书里的经典例子（Jason Wei 的 CoT 原论文）：

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: The answer is 11.   ← one-shot 给的示范答案

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
   bought 6 more, how many apples do they have?
A: The answer is 27.   ← 错
```

加入中间推理后：

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is
   6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
   bought 6 more, how many apples do they have?
A: The cafeteria started with 23 apples. They used 20 to make lunch,
   so they had 23 - 20 = 3. Then they bought 6 more, so they now
   have 3 + 6 = 9. The answer is 9.   ← 对
```

### Zero-shot CoT vs Few-shot CoT

CoT 有两种触发方式：

```python
# Few-shot CoT：提供带推理过程的示范
cot_prompt = [
    {"role": "user", "content": "Roger has 5 tennis balls..."},
    {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 ... The answer is 11."},
    {"role": "user", "content": "The cafeteria had 23 apples..."},
]

# Zero-shot CoT：一句话咒语
zero_cot = [{
    "role": "user",
    "content": "The cafeteria had 23 apples... Let's think step-by-step."
}]
```

"Let's think step by step" 来自 Kojima et al. (2022) 的 *"Large Language Models are Zero-Shot Reasoners"*。类似效果的变体还有："Take a deep breath and think step-by-step"、"Let's work through this problem step-by-step"。

对小模型（< 100B），**few-shot CoT 通常显著优于 zero-shot CoT**；对大模型（GPT-4/Claude 级别），两者差距变小。

## 推理范式 II：Self-Consistency（自洽投票）

### 原理：多次采样 + 多数投票

CoT 还有一个脆弱点——如果中间某一步推错了，最终答案就错。Self-Consistency 的思路很朴素：

> 同一个 prompt 跑 N 次（`temperature > 0` 引入多样性），收集 N 条不同的推理链，最后对**最终答案**做多数投票。

![Self-Consistency：三条推理链投票，多数得 11 正确](/book-notes/hands-on-llm/images/prompt-selfconsistency.png)

想象三次采样：

- 推理链 A：`2×3 = 6 tennis balls` → **答案 6**（错，漏掉 Roger 原有的 5 个）
- 推理链 B：`5 + 6 = 11` → **答案 11**（对）
- 推理链 C：`2 cans × 3 = 6, plus 5 = 11` → **答案 11**（对）

投票：**11 胜**。一次错误采样被两次正确采样淹没了。

### 适用与不适用

**适合**：

- 数学、逻辑这类有**唯一正确答案**的任务。
- 代码生成（可以用测试用例做"自动投票"）。
- 多选题。

**不适合**：

- 开放式生成（小说、摘要）——"多数票"没法定义。
- 对延迟敏感的实时对话。

代码上非常简单：

```python
from collections import Counter

answers = []
for _ in range(5):
    out = pipe(zero_cot_prompt, do_sample=True, temperature=0.7)
    answer = extract_final_answer(out[0]["generated_text"])
    answers.append(answer)

final = Counter(answers).most_common(1)[0][0]
```

**代价**：推理成本 N 倍。典型工程里取 N = 5–10，性价比最好。

## 推理范式 III：Tree-of-Thought（ToT）

Self-Consistency 是"多次独立采样再投票"——每条推理链之间没有交流。ToT 更进一步：**每一步都生成多个候选，择优保留，再展开下一步**，相当于在推理步骤上做 BFS/DFS 搜索。

![Tree-of-Thought：每步多候选，评估剪枝再展开](/book-notes/hands-on-llm/images/prompt-tot.png)

结构上：

- **State**：当前已完成的推理片段。
- **Action**：生成若干可能的下一步思考。
- **Evaluator**：对候选打分（让 LLM 自己打，或者用规则/外部工具）。
- **Search**：保留 top-k 展开，剪掉低分分支。

### 原始 ToT vs 简化版 ToT Prompt

原始 Tree-of-Thought（Yao et al., 2023）需要多次调用 LLM 构建搜索树，工程复杂度高。书里介绍了一种**用 prompt 模拟 ToT** 的简化做法——让模型在一次对话里扮演多位专家，互相讨论：

```python
zeroshot_tot_prompt = [{
    "role": "user",
    "content": (
        "Imagine three different experts are answering this question. "
        "All experts will write down 1 step of their thinking, then share "
        "it with the group. Then all experts will go on to the next step, "
        "etc. If any expert realizes they're wrong at any point then they leave. "
        "The question is 'The cafeteria had 23 apples. If they used 20 to "
        "make lunch and bought 6 more, how many do they have?' "
        "Make sure to discuss the results."
    )
}]
```

这种 prompt-only ToT 成本远低于真正的树搜索，效果却能接近一部分完整 ToT 的收益，很适合生产环境。

### 什么任务适合 ToT

- **需要规划的任务**：写作大纲、策略决策、多步骤编程。
- **存在"走错一步就翻车"的搜索问题**：Game of 24、Crosswords、Sudoku。
- **允许较高延迟和成本**的场景。

## 三种推理范式对比

下面这个交互式可视化用同一道算术题（23 apples）演示四种解法的"思考轨迹"和预估成本：

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/reasoning-strategies.html" height="660px" title="四种推理策略对比" />

| 方法 | API 调用数 | 典型准确率提升 | 实现复杂度 | 适用场景 |
|------|-----------|----------------|-----------|---------|
| Standard Prompt | 1 | — | 极低 | 简单问答、文本改写 |
| Chain-of-Thought | 1 | +10–30% | 低 | 数学、逻辑、多步推理 |
| Self-Consistency | N (5–40) | +5–15% 相对 CoT | 中 | 有唯一答案的任务 |
| Tree-of-Thought | N × 树深度（真 ToT）/ 1（prompt-only） | +10–20% 相对 CoT | 高 | 规划、搜索、创意探索 |

## 输出可靠性：怎么保证输出能用

推理做对了还不够——生产系统里**输出必须是合法的 JSON / SQL / 受限集合里的某个词**，不然下游 parser 直接崩。

作者把输出约束分成三类：

1. **Examples（示例约束）**：用 few-shot 给出期望格式的样本。
2. **Grammar（受约束采样）**：在解码时强制只能选合法 token。
3. **Fine-tuning（微调）**：训练一个天然输出正确格式的模型（放到第 12 章讲）。

### Examples：用 few-shot 约束格式

让模型输出 RPG 角色 JSON：

```python
# Zero-shot：很可能输出不合法 JSON
zero_prompt = [{"role": "user", "content": "Create a character profile for an RPG game in JSON format."}]
# 典型输出：带 ```json 代码块、字段不完整、最后被 max_tokens 截断

# One-shot：给一个结构模板
one_shot_template = """Create a short character profile for an RPG game. Make sure to only use this format:

{
  "description": "A SHORT DESCRIPTION",
  "name": "THE CHARACTER'S NAME",
  "armor": "ONE PIECE OF ARMOR",
  "weapon": "ONE OR MORE WEAPONS"
}
"""
```

给了模板之后，模型 90% 情况会严格照搬结构——但**仍有破坏格式的可能**。

### Grammar：真正强制的受约束采样

终极方案是在**解码阶段**就只允许合法的 token。具体实现：

- **GBNF（GGML BNF）**：`llama.cpp` / `llama-cpp-python` 支持的语法规范，可以定义正则、JSON schema 甚至完整 SQL 语法。
- **Outlines、Guidance、Guardrails、LMQL**：Python 生态里常用的约束库。
- **OpenAI `response_format={"type": "json_object"}`**：商业 API 的内置 JSON mode。

用 `llama-cpp-python` 的 JSON 约束：

```python
from llama_cpp.llama import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*fp16.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Create a warrior for an RPG in JSON format."}],
    response_format={"type": "json_object"},   # 内部转成 GBNF 强制 JSON
    temperature=0,
)["choices"][0]["message"]["content"]

import json
json_output = json.dumps(json.loads(output), indent=4)  # 100% 保证合法
```

**工作原理**：在每个采样步骤，解码器根据当前已生成的前缀和语法规则，把**非法 token 的 logits 设为 -∞**，再做 softmax + 采样。换句话说，语法规则"物理上"阻止了模型输出错误格式。

![受约束采样：只允许 positive/neutral/negative 三个 token](/book-notes/hands-on-llm/images/prompt-constrained.png)

典型用法：

- **分类任务**：强制输出必须是 `{positive, neutral, negative}` 之一。
- **函数调用 / 工具使用**：输出必须符合某个 JSON schema。
- **SQL 生成**：强制符合 SQL 语法（避免无效关键字）。
- **代码生成**：绑定目标语言的 AST。

::: warning 注意
受约束采样**不会让模型答对**——它只保证格式合法。语义错误（比如把 positive 说成 negative）仍然要靠 prompt engineering + 更好的模型来解决。
:::

## 小结：Prompt 工程的能力金字塔

从底层到高层，Prompt 工程的技能点大致是这样一座塔：

| 层级 | 能力 | 典型工具 |
|------|------|---------|
| L1：基础 | 写清楚指令、知道参数怎么调 | `temperature`、`top_p`、`max_tokens`、chat template |
| L2：结构化 | 七要素组合、few-shot 示范 | Persona + Instruction + Examples |
| L3：任务分解 | 把复杂任务拆成 prompt 链 | Chain prompting、LangChain |
| L4：激发推理 | 让模型在 System 2 模式思考 | CoT、"Let's think step by step" |
| L5：采样与搜索 | 用成本换准确率 | Self-Consistency、ToT、Best-of-N |
| L6：强约束输出 | 保证输出 100% 合法 | GBNF、Outlines、Guidance、JSON mode |

**几条可以带走的准则**：

- **小模型值得认真调 prompt**，大模型值得认真调 CoT。
- **示例永远比描述更强**——"说明文"加一个例子胜过三段形容词。
- **CoT 不是银弹**——对极简单任务反而会让输出变啰嗦，延迟变高。
- **受约束采样 > few-shot 格式约束**——能用 grammar 就别靠模型自觉。
- **Self-Consistency 是"用钱买准确率"最直接的旋钮**。

下一章我们会看到，LangChain 其实就是把这一章里的 chain prompting、memory、工具调用做成标准化组件，从而让这些 pattern 从"一次性脚本"走向"可维护的应用"。
