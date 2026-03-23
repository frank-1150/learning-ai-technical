---
title: 智能体的原理和操控
description: AI 智能体的核心执行机制、工具调用循环，以及如何在运行时引导和控制智能体
tags: [agents, llm, tools, agent-loop]
---

# 智能体的原理和操控

AI 智能体不是"一次性 LLM 调用"，而是由宿主代码精确控制的循环系统。理解这个循环，是构建和调试任何 Agent 应用的基础。

## 文章

- [智能体的核心：两个循环](./agent-loop) — 从真实开源代码剖析驱动智能体的两个嵌套循环，以及 Steering / Follow-up Messages 的操控机制
