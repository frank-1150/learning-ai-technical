---
date: 2026-03-19
title: 智能体的核心：两个循环
description: 从真实开源代码剖析驱动 AI 智能体的两个嵌套循环，以及如何通过 Steering / Follow-up Messages 在运行时操控智能体
tags: [agents, agent-loop, tool-use, steering, llm]
---

# 智能体的核心：两个循环

> 本文基于 [badlogic/pi-mono agent-loop.ts](https://github.com/badlogic/pi-mono/blob/main/packages/agent/src/agent-loop.ts) 的真实实现，剖析驱动 AI 智能体的核心机制。

## 智能体不是"一次性调用"

很多人第一次接触 LLM 时，把它当成一个函数：输入 Prompt，输出文字。但智能体（Agent）不是这样工作的。

智能体的本质是一个**持续运行的循环**：LLM 生成回复 → 执行工具 → 把结果喂回 LLM → 再次生成 → 再次执行工具……直到任务完成。

这个循环不是"LLM 自动帮你做"的魔法，而是宿主代码（Host Code）精确控制的一套状态机。`agent-loop.ts` 里的 `runLoop()` 函数，就是这台状态机的核心引擎。

## 两个嵌套循环

`runLoop()` 的骨架非常简洁：两个嵌套的 `while` 循环。

```typescript
async function runLoop(currentContext, newMessages, config, ...) {
  let pendingMessages = await config.getSteeringMessages?.() || [];

  // 外层循环：处理 Follow-up Messages
  while (true) {
    let hasMoreToolCalls = true;

    // 内层循环：处理工具调用 + Steering Messages
    while (hasMoreToolCalls || pendingMessages.length > 0) {
      // 1. 注入 Steering Messages
      // 2. 调用 LLM 生成回复
      // 3. 执行工具调用（如果有）
      // 4. 重新检查 Steering Messages
    }

    // 检查 Follow-up Messages
    const followUp = await config.getFollowUpMessages?.() || [];
    if (followUp.length > 0) {
      pendingMessages = followUp;
      continue; // 重入外层循环
    }

    break; // 真正结束
  }
}
```

两个循环各有各的职责，理解它们是理解所有"智能体框架"的基础。

### 内层循环：工具调用循环

内层循环是智能体"干活"的地方。它的退出条件是：

- LLM 不再发出工具调用（`hasMoreToolCalls = false`）
- 没有待注入的 Steering Messages（`pendingMessages.length === 0`）

**每次迭代做三件事：**

1. **注入 Steering Messages**（如果有）— 把外部注入的消息先塞进上下文
2. **LLM 生成回复** — 调用 `streamAssistantResponse()`，流式接收输出
3. **执行工具调用** — 如果 LLM 回复中包含工具调用，执行它们，把结果加入上下文，然后回到第 1 步

这就是智能体"持续行动"的来源：只要 LLM 还在调用工具，内层循环就不会结束。

### 外层循环：跟进循环

外层循环的存在感很低，但它解决了一个重要问题：**Agent 完成一个任务后，怎么接受新任务？**

当内层循环退出后，外层循环会调用 `getFollowUpMessages()`，检查是否有新消息：

- **有** → 把新消息设为 `pendingMessages`，`continue` 重入外层循环（内层循环也会重新开始）
- **无** → `break`，整个 Agent 结束

这让外层代码可以在 Agent "准备停下来"的瞬间，把下一个任务塞给它，而不需要重启整个 Agent。

<HtmlVisualization src="/ai-applications/agents/visualizations/agent-loop.html" />

## 操控智能体：两种注入机制

`getSteeringMessages` 和 `getFollowUpMessages` 是两个 **callback**，由宿主代码提供。它们的区别是时机：

| | Steering Messages | Follow-up Messages |
|---|---|---|
| **时机** | Agent 运行期间，每次内层循环迭代时检查 | Agent 即将结束时检查 |
| **效果** | 在下一次 LLM 调用前注入 | 让 Agent 继续执行新任务 |
| **用途** | 实时引导、中途修正方向 | 追加新指令、链式任务 |

### Steering Messages：实时引导

设想一个场景：用户让 Agent 去搜索并写一篇报告。Agent 开始执行了，但用户突然想说"等等，只需要三段就够了"。

这条消息不应该等 Agent 完成后再说，而应该在下一次 LLM 调用前就注入进去。`getSteeringMessages` 就是这个入口：

```typescript
// 宿主代码维护一个消息队列
const steeringQueue: AgentMessage[] = [];

const config = {
  getSteeringMessages: async () => {
    const msgs = [...steeringQueue];
    steeringQueue.length = 0; // 取走后清空
    return msgs;
  }
};

// 用户任意时刻可以注入
steeringQueue.push({ role: 'user', content: '只需要三段' });
```

内层循环在每次迭代开始时都会调用一次 `getSteeringMessages()`：

```typescript
while (hasMoreToolCalls || pendingMessages.length > 0) {
  // 每次迭代顶部检查
  pendingMessages = (await config.getSteeringMessages?.()) || [];
  // ...
}
```

### Follow-up Messages：追加任务

`getFollowUpMessages` 则在 Agent "准备停下来"时才被调用，适合链式任务：

```typescript
const tasks = ['先搜索 X', '再总结成报告', '最后翻译成英文'];
let taskIndex = 0;

const config = {
  getFollowUpMessages: async () => {
    taskIndex++;
    if (taskIndex < tasks.length) {
      return [{ role: 'user', content: tasks[taskIndex] }];
    }
    return []; // 没有更多任务，Agent 真正结束
  }
};
```

## 工具执行的生命周期

每个工具调用不是直接运行的，而是经历一套完整的生命周期：

```
工具调用请求
    │
    ▼
① validateToolArguments   ← 参数格式校验
    │
    ▼
② beforeToolCall hook     ← 可以拦截并阻止执行
    │ (block: true → 立即返回错误)
    ▼
③ tool.execute            ← 实际执行（串行 or 并行）
    │                        执行期间持续发出 update 事件
    ▼
④ afterToolCall hook      ← 可以修改返回结果
    │
    ▼
ToolResultMessage         ← 结果加入上下文
```

### 串行 vs 并行

当 LLM 在一次回复中发出多个工具调用时，框架支持两种执行模式：

```typescript
if (config.toolExecution === "sequential") {
  // 一个一个执行，前一个完成才执行下一个
  return executeToolCallsSequential(...);
}
// 默认：并行执行所有工具调用
return executeToolCallsParallel(...);
```

并行模式下，所有工具调用同时启动（`Promise.all` 风格），适合互相独立的工具（如同时搜索多个关键词）。串行模式适合有依赖关系的工具（如先创建文件再写入）。

### beforeToolCall：人类在环（Human-in-the-Loop）

`beforeToolCall` hook 是最强大的控制入口之一。它可以在任意工具执行前介入：

```typescript
const config = {
  beforeToolCall: async ({ toolCall, args, context }) => {
    // 拦截危险操作
    if (toolCall.name === 'deleteFile') {
      const confirmed = await askUser(`确认删除 ${args.path}？`);
      if (!confirmed) {
        return { block: true, reason: '用户取消了删除操作' };
      }
    }
    // 返回 undefined 或 {} 表示允许继续执行
  }
};
```

这是构建"需要人类审批"的 Agent 系统的关键模式。

## 消息边界：AgentMessage vs LLM Message

`agent-loop.ts` 有一个重要的设计决策：**系统内部始终使用 `AgentMessage`，只在调用 LLM 时才转换为 `LLM Message[]`**。

```typescript
async function streamAssistantResponse(context, config, ...) {
  // 这是唯一的格式转换点
  const llmMessages = await config.convertToLlm(messages);

  const llmContext: Context = {
    systemPrompt: context.systemPrompt,
    messages: llmMessages, // LLM 格式
    tools: context.tools,
  };

  return streamFunction(config.model, llmContext, ...);
}
```

`AgentMessage` 比 LLM Message 更丰富，可以携带额外元数据（时间戳、工具执行详情等），而不受各家 LLM API 格式限制。转换只发生一次，边界清晰。

## 事件系统

整个循环运行期间，每一个关键节点都会发出事件：

```
agent_start
  turn_start
    message_start    ← Prompt 注入
    message_end
    message_start    ← LLM 开始生成
      message_update ← 流式 token
      message_update
      ...
    message_end      ← LLM 完成
    tool_execution_start
      tool_execution_update ← 工具进度
    tool_execution_end
    message_start    ← 工具结果
    message_end
  turn_end
agent_end
```

这套事件系统让宿主代码可以实现：实时流式 UI、日志记录、调试面板、取消/中断——而不需要侵入循环本身的逻辑。

## 小结

两个嵌套循环，加上两种消息注入机制，构成了现代 AI 智能体的骨架：

- **内层循环** — 驱动"工具调用 → 结果 → 再生成"的持续行动
- **外层循环** — 让 Agent 在任务间保持存活，接受链式指令
- **Steering Messages** — 运行中实时引导（人类在环的核心入口）
- **Follow-up Messages** — 任务结束后追加新指令
- **beforeToolCall hook** — 在执行前拦截，实现人类审批
- **事件系统** — 解耦循环逻辑与 UI/日志

理解了这两个循环，你也就理解了 Claude Code、Cursor、Devin 这些产品背后最核心的执行模型。
