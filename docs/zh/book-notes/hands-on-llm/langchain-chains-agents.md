---
date: 2026-04-17
title: "LangChain 三件套：Chains、Memory 与 ReAct Agent"
description: "从单次 prompt 到多步推理、从无记忆到能持续对话、从被动输出到主动调用工具——LangChain 的三大抽象如何把 LLM 从聊天框变成可用的系统组件。"
tags: [LangChain, Agent, ReAct, LLM 应用, Prompt Chain]
---

# LangChain 三件套：Chains、Memory 与 ReAct Agent

> 本文对应原书 **第 7 章 Advanced Text Generation Techniques and Tools**，覆盖：LangChain 的 Chain / Memory / Agent 三大抽象、ReAct 推理模式、量化模型加载。

## 为什么需要 LangChain

用 `transformers` 直接调 LLM，最小的交互单元是 **一次 generate**：给一段 prompt，拿一段输出。这足以写个问答 demo，但离"可用的系统组件"还差一大截——现实里几乎每一个生产级 LLM 应用都要解决至少下面这几件事：

- **Prompt 不是一次写死的字符串**，而是带变量的模板；同一个模板会在不同地方被复用。
- **任务常常无法用一个 prompt 完成**——摘要、翻译、抽取实体可能要串成三步。
- **对话需要记忆**——用户说过的名字、上下文偏好、几轮前的结论。
- **模型本身不会做数学、不会联网、不会查数据库**——它得知道自己"能调什么"，并且决定"什么时候调"。

如果每个项目都从零拼装这些东西，就会写一堆样板：prompt 渲染、历史拼接、工具调度、解析输出……LangChain 的核心价值，就是把这些重复的模式封装成四个可组合的抽象：

![LangChain 框架模块总览](/book-notes/hands-on-llm/images/langchain-framework-overview.png)

- **Model I/O**：统一 Prompts / LLMs / Output Parser 的接口，不管底层是 OpenAI 还是本地 GGUF。
- **Memory**：管理对话历史的三种经典策略。
- **Retrieval**：Embedding + 向量库 + 文档加载（放到下一章 RAG 讲）。
- **Agents**：把 LLM 当"决策引擎"，加上工具和 ReAct 循环。

本章沿着 **单链 → 多链 → 加记忆 → 加工具** 的路径，把这四块拼成一个能自己推理、会调用搜索和计算器的 agent。

::: tip 时代注脚
LangChain 是最早把这些概念标准化的框架。现在更"原生"的方式是 **DSPy**（把 prompt 当作可优化的程序）、**Haystack**（搜索导向）、以及 **LangGraph / CrewAI**（显式的 agent 状态机）。思路一脉相承——Chain 是 DAG、Memory 是 State、Agent 是 Controller，换个 SDK 核心概念不变。
:::

---

## Model I/O：加载量化模型

在把 LLM 接入 chain 之前，先要"加载 LLM"。书中选了 **Phi-3-mini-4k-instruct** 的 **GGUF** 格式——这里有两个关键词：

- **GGUF** 是 `llama.cpp` 生态的模型文件格式，专门为 CPU / 混合 GPU 推理优化。
- **量化（Quantization）** 把原本 16-bit 或 32-bit 的权重压成更少的位数（4-bit / 8-bit）。代价是精度略降，收益是显存减半、速度更快。

一个直观的类比：问"现在几点"，你可以说 14:16:12（FP32），也可以说 14:16（FP16），还可以说 "两点一刻"（INT4）——信息越粗，传达成本越低，但足够应付大多数日常场景。

::: tip 量化的实用底线
书里给出的经验法则：**至少选 4-bit 量化**。再往下（3-bit / 2-bit）虽然能跑，但质量损失会明显到影响使用——与其硬压，不如换个更小但精度更高的模型。
:::

下载模型：

```bash
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
```

用 LangChain 的 `LlamaCpp` 封装加载：

```python
from langchain import LlamaCpp

llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,   # -1 表示尽可能把所有层都卸载到 GPU
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False,
)

# 统一的调用接口
llm.invoke("Hi! My name is Maarten. What is 1 + 1?")
```

不管底层是本地 GGUF、OpenAI API、还是 Anthropic API，LangChain 的接口都统一成 `llm.invoke(...)`。想切成 ChatGPT，只要换一行：

```python
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI(openai_api_key="MY_KEY")
```

这种**运行时可替换**就是框架存在的第一个理由：开发时本地跑，生产上换成 API，业务代码一个字不改。

---

## Chains：把 LLM 串成流水线

LangChain 名字的来源就是 **chain**——一个 chain 把 LLM 和别的组件（prompt 模板、工具、外部存储）连起来，让一次调用变成可复用的流水线节点。

### 单链：PromptTemplate + LLM

最简单的 chain 只有两个节点：**prompt 模板 + LLM**。直接用 `llm.invoke("Hi!")` 看似能工作，但问题在于 **Phi-3 这种 instruct 模型期待特殊的 chat 模板**——缺了 `<|user|> / <|assistant|>` 这类控制 token，模型根本不知道要回答什么，输出会是空串。

解决办法：把这套模板写成 `PromptTemplate`，然后用管道符 `|` 串起来：

```python
from langchain import PromptTemplate

# Phi-3 的专用 chat 模板
template = """<s><|user|>
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"],
)

# 一个 chain = prompt | llm
basic_chain = prompt | llm

basic_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
# -> "The answer to 1 + 1 is 2. It's a basic arithmetic operation..."
```

这行 `prompt | llm` 是 **LangChain Expression Language（LCEL）** 的核心语法——`|` 左边的输出直接喂给右边的输入。概念上和 Unix 管道一致。

单链的结构就这么简单：

![单链结构：模板组件 + LLM](/book-notes/hands-on-llm/images/langchain-chain-example.png)

PromptTemplate 还可以承担变量替换，比如：

```python
template = "Create a funny name for a business that sells {product}."
name_prompt = PromptTemplate(template=template, input_variables=["product"])
chain = name_prompt | llm

chain.invoke({"product": "coffee"})
```

### 多链：把复杂任务拆成若干个 prompt

一个足够好的 prompt 能一次搞定简单任务，但遇到"生成一个完整故事"这种复合目标就不行了——输出要同时包含**标题**、**主角描述**、**完整剧情**，一次全塞给模型，模型要么漏、要么各部分风格不一致。

更可靠的做法是把任务拆成若干子 prompt、依次执行，前一步的输出作为后一步的输入：

![多 prompt 串联：Title → Character → Story](/book-notes/hands-on-llm/images/langchain-multi-prompt-chain.png)

代码层面，用 `LLMChain` 封装每一段，然后用 `|` 把三段连起来：

```python
from langchain import LLMChain

# --- 第一段：根据摘要生成标题 ---
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title = LLMChain(llm=llm, prompt=title_prompt, output_key="title")

# --- 第二段：根据摘要+标题生成主角描述 ---
template = """<s><|user|>
Describe the main character of a story about {summary} with the title {title}.
Use only two sentences.<|end|>
<|assistant|>"""
character_prompt = PromptTemplate(
    template=template,
    input_variables=["summary", "title"],
)
character = LLMChain(llm=llm, prompt=character_prompt, output_key="character")

# --- 第三段：综合生成一段故事 ---
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main character is:
{character}. Only return the story and it cannot be longer than one paragraph.<|end|>
<|assistant|>"""
story_prompt = PromptTemplate(
    template=template,
    input_variables=["summary", "title", "character"],
)
story = LLMChain(llm=llm, prompt=story_prompt, output_key="story")

# --- 串起来 ---
llm_chain = title | character | story

llm_chain.invoke("a girl that lost her mother")
# -> {"summary": ..., "title": ..., "character": ..., "story": ...}
```

调用一次 `llm_chain.invoke(...)`，背后会发生**三次 LLM 调用**，最终字典里同时拿到标题、人物、故事——每个子任务只负责自己那部分，prompt 更短、更聚焦，模型出错的概率显著降低。

这是 LLM 工程里一条很实用的原则：**用多次小 prompt 替代一次大 prompt**。代价是延迟和 token 开销翻倍，但换来的是更好的可控性、更清晰的中间产物、以及（重要的）**独立可调试**——任何一步翻车，都能单独 replay 那一步。

---

## Memory：让 LLM 记住对话

LLM 本身是**无状态**的——这是架构层面的事实：每次 `forward()` 都只看当前 prompt，昨天说了什么、上一轮名字叫啥，模型一概不知。演示一下：

```python
# 先告诉模型我叫 Maarten
basic_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
# -> "Hello Maarten! The answer to 1 + 1 is 2."

# 再问它我叫什么
basic_chain.invoke({"input_prompt": "What is my name?"})
# -> "I don't have the ability to know personal information..."
```

模型不是"忘了"——它根本没有"记忆"这个概念。想让 LLM 看起来有记忆，只能把历史**显式地拼回 prompt 里**。

LangChain 把这件事封装成三种 memory 类型，处理的 trade-off 略有不同：

### Conversation Buffer：最朴素的全量历史

最直观的方式：每轮对话都把完整历史拼进 prompt：

```python
from langchain.memory import ConversationBufferMemory

template = """<s><|user|>Current conversation:{chat_history}

{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"],
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
llm_chain.invoke({"input_prompt": "What is my name?"})
# -> "Your name is Maarten."
```

这次它"记得"了。不过 buffer 的代价显而易见——**对话越长，prompt 越长**，到第 100 轮时每个 prompt 都把前面 99 轮完整搬进来，token 开销爆炸不说，很容易撞上 4K / 8K 的上下文上限。

### Windowed Conversation Buffer：只保留最近 N 轮

解决方式之一：**滑动窗口**，只保留最近 `k` 轮：

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

llm_chain.predict(input_prompt="Hi! My name is Maarten and I am 33 years old. What is 1+1?")
llm_chain.predict(input_prompt="What is 3 + 3?")
llm_chain.invoke({"input_prompt": "What is my name?"})
# -> "Your name is Maarten."  （第二轮还在窗口内）

llm_chain.invoke({"input_prompt": "What is my age?"})
# -> "I'm unable to determine your age..."  （年龄出现在第一轮，已经滑出窗口）
```

**prompt 大小恒定**（最多 `k * 每轮平均 token`），代价是超过窗口的信息就直接丢。

### Conversation Summary：滚动摘要

如果既要长程记忆、又要控制 prompt 规模，还有第三条路——**用另一次 LLM 调用把历史摘要出来**，只带摘要进下一轮 prompt：

```python
from langchain.memory import ConversationSummaryMemory

summary_prompt_template = """<s><|user|>Summarize the conversations and update
with the new lines.

Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""

summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template,
)

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    prompt=summary_prompt,
)

llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
```

这样每轮对话都有**两次 LLM 调用**：一次用主 prompt 回答用户，一次用 summary prompt 更新摘要。优点是历史再长也能压缩成几百 token；缺点也明显——**摘要质量依赖小 LLM**，而且每轮都要额外跑一次模型，延迟翻倍。

::: warning 摘要不是免费的
摘要会**丢细节**。书里的例子：先问"1+1 是多少"，过几轮再问"我问的第一个问题是什么"——如果那次具体的算式没进摘要，模型只能推断，未必准确。**需要被精确记住的信息不能指望摘要兜底**——要么在 prompt 外单独记在结构化状态里，要么明确 instruction 让摘要必须保留原始问句。
:::

### 三种策略的对比

![Memory 三种策略的利弊对比（原书 Table 7-1）](/book-notes/hands-on-llm/images/langchain-memory-comparison.png)

整理成工程视角的对照表：

| 维度 | ConversationBuffer | ConversationBufferWindow | ConversationSummary |
|------|---|---|---|
| **信息完整性** | 100%（上下文内） | 最近 k 轮 100%，更早全丢 | 摘要后有损 |
| **Prompt 增长** | 线性随轮数 | 恒定（与 k 成正比） | 亚线性（摘要压缩） |
| **每轮 LLM 调用** | 1 次 | 1 次 | 2 次（回答 + 摘要） |
| **延迟** | 低 | 最低 | 最高 |
| **Token 成本** | 最高 | 最低 | 中等 |
| **适合场景** | 短对话（<20 轮） | 交易型、指令型任务 | 长对话助手、教练式应用 |
| **典型坑** | 撞上下文上限 | 关键信息滑出窗口 | 摘要丢关键细节 |

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/memory-tradeoff.html" height="450px" title="三种 Memory 策略的 trade-off" />

拖动轮数滑块，直观地看三种策略的 token 消耗、信息保留率、延迟怎么随对话长度发散——这基本上决定了实际选型。

::: tip 现代实践里更常见的组合策略
生产系统里很少单用一种 memory。常见做法是：**最近 3-5 轮用 buffer**（保留精确细节）+ **更早的部分跑 summary**（保留上下文）+ **关键字段（用户名、偏好、订单号）单独存进结构化记忆**——三层叠加，成本和信息完整性都可控。LangChain 的 `CombinedMemory` 就是为此而生。
:::

---

## Agents：让 LLM 变成决策引擎

前面的 chain 都是**固定流程**——流程图是我们画的，模型只负责填空。Agent 把这件事倒过来：**给模型一组工具，让模型自己决定用哪个、什么时候用、用几次**。

### 为什么需要工具

LLM 的短板很清楚——算不对数、查不到当前信息、不能操作外部系统。书里用一个最直白的例子演示这种差异：

![LLM 加上工具之后，能从"猜答案"变成"算答案"](/book-notes/hands-on-llm/images/langchain-agent-tools.png)

直接问"47 / 12 × 3.14 等于几"，LLM 靠模式匹配硬猜，经常算错。但如果告诉它"你可以调用 `calculator` 工具"，它就会判断出"这是数学题，得用计算器"，把表达式传过去，拿结果再给用户——瞬间从 7.34（错）变成 12.2983（对）。

Agent 的能力来自两个组件：
- **Tools**：LLM 自己做不到但可以调用的能力（搜索、计算器、数据库、shell……）。
- **Agent Type**：决定"怎么推理 / 什么时候调工具"的策略，其中最经典的是 **ReAct**。

### ReAct：Reasoning + Acting

ReAct（Yao et al. 2022，arXiv:2210.03629）是现在几乎所有 agent 框架的思想原型。核心很简单：把推理和行动**交错**进行，强制模型用三元组在每一步表达自己：

- **Thought**：我现在该做什么、为什么。
- **Action**：调用哪个工具、传什么参数。
- **Observation**：工具返回了什么。

然后 Thought → Action → Observation → Thought → Action → … 一直循环，直到模型在 Thought 里说"我已经知道答案了"，输出 **Final Answer**。

ReAct 的 prompt 长这样：

![ReAct prompt 模板（原书 Figure 7-15）](/book-notes/hands-on-llm/images/langchain-react-template.png)

模板告诉模型：**只能按 Thought / Action / Observation 的格式输出**，并列出所有可用工具。LangChain 的实际模板：

```text
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

模型在自己的"草稿本"（`agent_scratchpad`）里累积 Thought/Action/Observation，每轮新的 Thought 都能看到前面所有痕迹。

一个典型的两轮 ReAct 循环：

![ReAct 两轮循环示意（原书 Figure 7-16）](/book-notes/hands-on-llm/images/langchain-react-cycles.png)

问题："MacBook Pro 现在多少美元？按 0.85 的汇率换算成欧元是多少？"

- **Cycle 1**：Thought "我该先搜网"→ Action `Google[price MacBook Pro]` → Observation "1,299 美元"。
- **Cycle 2**：Thought "现在需要计算器" → Action `Calculator[1299 × 0.85]` → Observation "1104.15"。
- Thought "我知道最终答案了" → Final Answer。

每一轮都是**一次完整的 LLM 调用**——所以 ReAct 的成本是"轮数 × 单次 prompt 长度"，很容易累加。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/react-agent-loop.html" height="650px" title="ReAct Agent 交互式演示" />

上面这个可视化可以切换几个预设场景，手动按"下一步"看每一次 Thought / Action / Observation 的推进，顺便看 LLM 调用次数和 token 开销怎么累积。

### LangChain 里写一个 ReAct Agent

书里用的例子：一个能搜索 + 会计算的 agent。因为 ReAct 对模型的指令遵循能力要求较高，本地的 Phi-3 不太够用，这里换到 OpenAI `gpt-3.5-turbo`：

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "MY_KEY"
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

然后定义 ReAct prompt 模板（就是上面那一段）：

```python
react_template = """Answer the following questions as best you can. You have
access to the following tools:

{tools}

Use the following format:
...
Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
)
```

注册工具——DuckDuckGo 搜索 + 基于 LLM 的数学计算器：

```python
from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search engine for general queries.",
    func=search.run,
)

tools = load_tools(["llm-math"], llm=openai_llm)
tools.append(search_tool)
```

最后把 ReAct agent 和 `AgentExecutor` 拼起来——`AgentExecutor` 是真正负责 **驱动循环** 的对象：

```python
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(openai_llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,             # 打印中间步骤
    handle_parsing_errors=True,
)

agent_executor.invoke({
    "input": "What is the current price of a MacBook Pro in USD? "
             "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD."
})
```

`verbose=True` 打开后，控制台会打出完整的 Thought / Action / Observation 流水线：

```text
> Entering new AgentExecutor chain...
I need to find the current price of MacBook Pro in USD first before converting it to EUR.
Action: duckduck
Action Input: "current price MacBook Pro in USD"[snippet: View at Best Buy. The best MacB...]
Action: Calculator
Action Input: $2,249.00 * 0.85
Answer: 1911.6499999999999
I now know the final answer
Final Answer: The current price of a MacBook Pro in USD is $2,249.00. It would
cost approximately 1911.65 EUR with an exchange rate of 0.85 EUR for 1 USD.
```

### ReAct 的局限

这套机制很强大，但不是银弹：

1. **每步一次 LLM 调用**。一个 3 轮 ReAct agent = 3-4 次 LLM 调用，成本和延迟线性累加。这也是为什么生产里常把 agent 用于"只在必要时"的复杂问题、而不是每条消息都走 agent。
2. **容易进死循环**。模型输出格式一旦跑偏（比如 Thought 后面没跟 Action），解析器报错——`handle_parsing_errors=True` 只是兜底，不能彻底解决。
3. **对模型能力要求高**。书中明确说 Phi-3 不够用——小模型的指令遵循常常不稳，Thought/Action 格式会被破坏。
4. **没有 human-in-the-loop**。agent 跑完给出答案，中间步骤对不对没人校验。严肃场景要么做 step-by-step 确认，要么要求 agent 输出引用来源（比如搜索返回的 URL），方便事后核查。

::: warning 从 LangChain 到现代 Agent 框架
LangChain 的 `AgentExecutor` 本质是一个隐式的循环，开发者很难精细控制状态。这也是 **LangGraph** 出现的原因——显式地把 agent 画成**状态机（节点 + 边）**，每个 Thought/Action/Observation 都是一个节点，边就是"下一步跳哪儿"。**CrewAI / AutoGen** 则走**多 agent 协作**的路线：一个团队里有 Researcher / Writer / Critic，各自负责一部分。底层还是 ReAct 或其变体——改变的是编排方式和可观测性。
:::

---

## 小结

| 抽象 | 解决的问题 | 最典型的用法 | 主要代价 |
|------|-----------|-------------|---------|
| **Chain** | 把 LLM 和组件拼成可复用流水线 | PromptTemplate \| LLM \| OutputParser | 增加框架依赖 |
| **多链** | 复杂任务拆成多步 | Title → Character → Story | 延迟 × N，token × N |
| **Buffer Memory** | 短对话保留精确历史 | 客服、FAQ | 撞上下文上限 |
| **Window Memory** | 控制 prompt 恒定大小 | 交易型 chatbot | 关键信息滑出 |
| **Summary Memory** | 长对话压缩历史 | 教练、陪伴式应用 | 每轮多一次 LLM 调用 |
| **ReAct Agent** | LLM 动态决定调用工具 | 搜索+计算、查 API、操作系统 | 每步一次 LLM，容易发散 |

这一章可以串成一条清晰的升级路径：**单次 prompt → 带模板的单链 → 多链流水线 → 加记忆的对话链 → 会调工具的 agent**。每一步都是在前一步上叠加新的抽象，换来新的能力，也带来新的成本。实际做应用时的判断标准几乎是反向的——**先问"最简单的那层够不够"，不够再往上加**。很多场景其实一个带 prompt template 的单链就够用了，没必要一上来就往 agent 上怼。

下一章会把"检索"这块从 LangChain 的 Retrieval 模块展开——那才是 LLM 真正能大量接入知识的入口：RAG。
