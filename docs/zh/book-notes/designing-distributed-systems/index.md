---
date: 2026-04-10
title: "Designing Distributed Systems：模式、范式与 Kubernetes 实践"
description: Brendan Burns《Designing Distributed Systems》第二版阅读笔记总览
tags: [distributed-systems, kubernetes, design-patterns, book-notes]
---

# Designing Distributed Systems：模式、范式与 Kubernetes 实践

> **Designing Distributed Systems: Patterns and Paradigms for Scalable, Reliable Systems Using Kubernetes**, 2nd Edition, Brendan Burns, O'Reilly 2024

Brendan Burns 是 Kubernetes 的联合创始人、微软公司副总裁，负责 Azure 管理与治理、Azure Arc、Kubernetes on Azure 等核心产品线。这本书是他将二十余年分布式系统工程经验凝练为可复用设计模式的尝试。

## 为什么读这本书

面向对象编程有 GoF 的 23 种设计模式，算法有 Knuth 的 *The Art of Computer Programming*。但分布式系统长期缺少这样一套 **共享语言**——工程师们反复在不同项目中重新发明 Sidecar、Ambassador、Scatter/Gather 这些模式，却没有统一的术语来讨论它们。

这本书试图填补这个空白。它不是一本 Kubernetes 操作手册，而是借助容器和编排平台，将分布式系统中反复出现的架构模式分类、命名、文档化。读完之后，你获得的不是某个具体技术的使用方法，而是一套 **思考分布式系统的框架**。

## 全书结构

<HtmlVisualization src="/book-notes/designing-distributed-systems/visualizations/book-structure.html" height="520px" title="全书结构总览" />

本书分为五个部分，从基础概念到单节点、多节点、批处理，最后到跨模式的通用原则：

### [Part I — 基础概念](./foundational-concepts)
**Chapter 1-2** | 软件模式的演进史 + 分布式系统的核心原语

从算法形式化 → OOP 设计模式 → 开源软件的历史脉络出发，解释为什么分布式系统也需要设计模式。然后介绍 API、延迟、可靠性、幂等性、一致性等构建分布式系统的基础概念。

### [Part II — 单节点模式](./single-node-patterns)
**Chapter 3-5** | Sidecar · Ambassador · Adapter

在单台机器（一个 Pod）内，通过多容器组合实现关注点分离。三种模式分别解决：扩展功能（Sidecar）、代理外部服务（Ambassador）、标准化接口（Adapter）。

### [Part III — 服务模式](./serving-patterns)
**Chapter 6-10** | 负载均衡 · 分片 · Scatter/Gather · FaaS · Leader Election

跨越多个节点的在线服务架构。从最简单的无状态副本，到一致性哈希分片、Scatter/Gather 并行查询、事件驱动的 FaaS，再到分布式锁和 Leader 选举。

### [Part IV — 批处理计算模式](./batch-computational-patterns)
**Chapter 11-13** | 工作队列 · 事件驱动批处理 · 协调批处理

大规模数据处理的三种编排范式。工作队列模式将任务生产与消费解耦；事件驱动模式用 Copier/Filter/Splitter/Merger 构建流水线；协调批处理用 Join/Reduce 实现 MapReduce 式的并行计算。

### [Part V — 通用概念](./universal-concepts)
**Chapter 14-16** | 可观测性 · AI 推理服务 · 常见故障模式

跨越所有模式的通用原则。如何建设可观测性体系（日志、指标、追踪）；如何在分布式环境中部署 AI 模型；以及九种在生产环境中反复出现的故障模式——这些是最难从教科书学到、只能从事故中总结的经验。
