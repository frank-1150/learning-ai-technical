---
date: 2026-03-19
title: "The Core of Agents: Two Loops"
description: Analyzing the two nested loops that drive AI agents from real open-source code, and how Steering / Follow-up Messages enable runtime control
tags: [agents, agent-loop, tool-use, steering, llm]
---

# The Core of Agents: Two Loops

> This article is based on the real implementation in [badlogic/pi-mono agent-loop.ts](https://github.com/badlogic/pi-mono/blob/main/packages/agent/src/agent-loop.ts), dissecting the core mechanism that drives AI agents.

## An Agent Is Not a "One-Shot Call"

Many people, when they first encounter an LLM, treat it like a function: send in a prompt, get back text. But agents don't work that way.

An agent is fundamentally a **continuously running loop**: the LLM generates a response → tools are executed → results are fed back to the LLM → the LLM generates again → tools are executed again… until the task is done.

This loop isn't magic that "the LLM does for you automatically" — it's a precisely controlled state machine managed by the host code. The `runLoop()` function in `agent-loop.ts` is the core engine of this state machine.

## Two Nested Loops

The skeleton of `runLoop()` is remarkably simple: two nested `while` loops.

```typescript
async function runLoop(currentContext, newMessages, config, ...) {
  let pendingMessages = await config.getSteeringMessages?.() || [];

  // Outer loop: handles Follow-up Messages
  while (true) {
    let hasMoreToolCalls = true;

    // Inner loop: handles tool calls + Steering Messages
    while (hasMoreToolCalls || pendingMessages.length > 0) {
      // 1. Inject Steering Messages
      // 2. Call LLM to generate a response
      // 3. Execute tool calls (if any)
      // 4. Re-check Steering Messages
    }

    // Check for Follow-up Messages
    const followUp = await config.getFollowUpMessages?.() || [];
    if (followUp.length > 0) {
      pendingMessages = followUp;
      continue; // Re-enter outer loop
    }

    break; // Truly done
  }
}
```

Each loop has its own responsibility, and understanding them is the foundation for understanding every agent framework.

### Inner Loop: The Tool-Call Loop

The inner loop is where the agent does its work. It exits when:

- The LLM stops issuing tool calls (`hasMoreToolCalls = false`)
- There are no pending Steering Messages (`pendingMessages.length === 0`)

**Each iteration does three things:**

1. **Inject Steering Messages** (if any) — push externally injected messages into context first
2. **LLM generates a response** — call `streamAssistantResponse()`, receive output as a stream
3. **Execute tool calls** — if the LLM response contains tool calls, execute them, add results to context, then return to step 1

This is the source of the agent's "continuous action": as long as the LLM keeps calling tools, the inner loop keeps running.

### Outer Loop: The Follow-Up Loop

The outer loop has a low profile, but it solves an important problem: **once an agent finishes one task, how does it accept the next one?**

When the inner loop exits, the outer loop calls `getFollowUpMessages()` to check for new messages:

- **Found** → set new messages as `pendingMessages`, `continue` to re-enter the outer loop (the inner loop also restarts)
- **None** → `break`, the entire agent ends

This lets the outer code inject the next task at the exact moment the agent "is about to stop," without needing to restart the whole agent.

<HtmlVisualization src="/ai-applications/agents/visualizations/agent-loop.html" />

## Steering the Agent: Two Injection Mechanisms

`getSteeringMessages` and `getFollowUpMessages` are two **callbacks** provided by the host code. The difference is timing:

| | Steering Messages | Follow-up Messages |
|---|---|---|
| **When** | During agent execution, checked at each inner loop iteration | Checked when the agent is about to stop |
| **Effect** | Injected before the next LLM call | Causes the agent to continue with a new task |
| **Use case** | Real-time guidance, mid-course correction | Appending new instructions, chained tasks |

### Steering Messages: Real-Time Guidance

Imagine this scenario: a user asks an agent to search for something and write a report. The agent starts executing, but the user suddenly wants to say "actually, just three paragraphs is enough."

This message shouldn't wait until the agent finishes — it should be injected before the next LLM call. `getSteeringMessages` is that entry point:

```typescript
// Host code maintains a message queue
const steeringQueue: AgentMessage[] = [];

const config = {
  getSteeringMessages: async () => {
    const msgs = [...steeringQueue];
    steeringQueue.length = 0; // Clear after consuming
    return msgs;
  }
};

// User can inject at any time
steeringQueue.push({ role: 'user', content: 'Just three paragraphs' });
```

The inner loop calls `getSteeringMessages()` at the top of every iteration:

```typescript
while (hasMoreToolCalls || pendingMessages.length > 0) {
  // Check at the top of each iteration
  pendingMessages = (await config.getSteeringMessages?.()) || [];
  // ...
}
```

### Follow-up Messages: Chaining Tasks

`getFollowUpMessages` is only called when the agent "is about to stop," making it ideal for chained tasks:

```typescript
const tasks = ['First search for X', 'Then summarize into a report', 'Finally translate to Chinese'];
let taskIndex = 0;

const config = {
  getFollowUpMessages: async () => {
    taskIndex++;
    if (taskIndex < tasks.length) {
      return [{ role: 'user', content: tasks[taskIndex] }];
    }
    return []; // No more tasks, agent truly ends
  }
};
```

## The Tool Execution Lifecycle

Every tool call doesn't just run directly — it goes through a complete lifecycle:

```
Tool call request
    │
    ▼
① validateToolArguments   ← Validate argument format
    │
    ▼
② beforeToolCall hook     ← Can intercept and block execution
    │ (block: true → return error immediately)
    ▼
③ tool.execute            ← Actual execution (sequential or parallel)
    │                        Emits update events throughout
    ▼
④ afterToolCall hook      ← Can modify the return value
    │
    ▼
ToolResultMessage         ← Result added to context
```

### Sequential vs. Parallel

When the LLM issues multiple tool calls in a single response, the framework supports two execution modes:

```typescript
if (config.toolExecution === "sequential") {
  // Execute one at a time, each waits for the previous to finish
  return executeToolCallsSequential(...);
}
// Default: execute all tool calls in parallel
return executeToolCallsParallel(...);
```

In parallel mode, all tool calls start simultaneously (`Promise.all` style) — ideal for independent tools (e.g., searching multiple keywords at once). Sequential mode is for tools with dependencies (e.g., create a file first, then write to it).

### beforeToolCall: Human-in-the-Loop

The `beforeToolCall` hook is one of the most powerful control points. It can intervene before any tool executes:

```typescript
const config = {
  beforeToolCall: async ({ toolCall, args, context }) => {
    // Intercept dangerous operations
    if (toolCall.name === 'deleteFile') {
      const confirmed = await askUser(`Confirm deletion of ${args.path}?`);
      if (!confirmed) {
        return { block: true, reason: 'User cancelled the delete operation' };
      }
    }
    // Return undefined or {} to allow execution to proceed
  }
};
```

This is the key pattern for building agent systems that require human approval.

## Message Boundaries: AgentMessage vs. LLM Message

`agent-loop.ts` makes an important design decision: **the system always uses `AgentMessage` internally and only converts to `LLM Message[]` when calling the LLM.**

```typescript
async function streamAssistantResponse(context, config, ...) {
  // This is the single format conversion point
  const llmMessages = await config.convertToLlm(messages);

  const llmContext: Context = {
    systemPrompt: context.systemPrompt,
    messages: llmMessages, // LLM format
    tools: context.tools,
  };

  return streamFunction(config.model, llmContext, ...);
}
```

`AgentMessage` is richer than LLM Message — it can carry extra metadata (timestamps, tool execution details, etc.) without being constrained by the API format of any particular LLM provider. The conversion happens once, at a clear boundary.

## The Event System

Throughout the loop's execution, every critical node emits an event:

```
agent_start
  turn_start
    message_start    ← Prompt injected
    message_end
    message_start    ← LLM starts generating
      message_update ← Streaming tokens
      message_update
      ...
    message_end      ← LLM finishes
    tool_execution_start
      tool_execution_update ← Tool progress
    tool_execution_end
    message_start    ← Tool result
    message_end
  turn_end
agent_end
```

This event system lets host code implement real-time streaming UIs, logging, debug panels, and cancellation/interruption — without touching the loop's own logic.

## Summary

Two nested loops, plus two message injection mechanisms, form the skeleton of a modern AI agent:

- **Inner loop** — drives the continuous action cycle of "tool call → result → generate again"
- **Outer loop** — keeps the agent alive between tasks, accepting chained instructions
- **Steering Messages** — real-time guidance during execution (the core entry point for human-in-the-loop)
- **Follow-up Messages** — append new instructions after a task completes
- **beforeToolCall hook** — intercept before execution to implement human approval
- **Event system** — decouples loop logic from UI and logging

Understand these two loops, and you understand the core execution model behind Claude Code, Cursor, Devin, and every other agentic product.
