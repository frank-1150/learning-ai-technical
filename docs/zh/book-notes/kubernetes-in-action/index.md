---
title: "Kubernetes in Action 阅读笔记"
description: "Marko Lukša 著，Manning 出版。从容器基础到生产实践，系统拆解 Kubernetes 18 章核心内容。"
date: "2026-04-10"
---

# Kubernetes in Action 阅读笔记

## 书籍简介

**Kubernetes in Action** 由 Marko Lukša 撰写，Manning 出版社 2018 年出版（ISBN: 9781617293726）。Lukša 是 Red Hat 云赋能团队（Cloud Enablement）的软件工程师，自 2014 年底起深度参与 Kubernetes 和 OpenShift 的工程实践。

本书的写作契机颇具代表性：2014 年 Kubernetes 尚在 v1.0 发布之前，Lukša 所在团队需要将 Red Hat 的中间件产品迁移到 OpenShift/Kubernetes 平台上。在没有成熟文档的情况下摸索出的经验，最终沉淀为这本广受工程师认可的 Kubernetes 权威指南。

全书共 18 章，分三部分：
- **Part 1 Overview**（Ch1-2）：容器技术演进与 K8s 初体验
- **Part 2 Core Concepts**（Ch3-10）：Pod、副本控制、服务、存储、配置、部署、有状态应用
- **Part 3 Beyond the Basics**（Ch11-18）：内部架构、安全、资源管理、调度、生产实践、扩展机制

书中所有示例均在 Kubernetes 1.8 上验证，配套源码与 YAML manifest 见 [github.com/luksa/kubernetes-in-action](https://github.com/luksa/kubernetes-in-action)。

---

## 为什么读这本书

Kubernetes 早已超越容器编排工具的范畴，成为现代云原生基础设施的操作系统。这一判断在 AI 时代尤为明显：

**云原生与 AI 基础设施的汇合点**。大模型训练集群（GPU 池）、推理服务（vLLM、TGI）、向量数据库、特征流水线——这些 AI 工程组件无一不运行在 Kubernetes 之上。理解 K8s 的调度、存储、网络模型，是构建可靠 AI 系统的前提，而不是可选项。

**声明式系统思维**。K8s 把"期望状态 vs. 实际状态"的 Reconciliation Loop 做到了极致，这套思维模型——describe what, not how——在分布式系统设计中有极强的迁移价值。

**故障排查的底层语言**。生产事故中，Pod OOMKill、调度 pending、网络策略误封、存储挂载失败……这些现象背后都需要对 K8s 内部机制有清晰认知。Lukša 的写法尤其注重"为什么这样设计"，读完后能显著提升 debug 效率。

**本书的独特价值**：相比官方文档的碎片化，本书提供了从容器基础到集群内部架构的完整心智模型；相比纯操作手册，它更关注原理。18 章读完，你对 K8s 的认知会从"会用 kubectl" 升级到"理解整个系统"。

---

## 全书文章总览

本系列共 9 篇阅读笔记，按内容聚合而非按章节简单拆分：

| 文章 | 覆盖章节 | 主题 |
|------|---------|------|
| [容器化与 K8s 全局观](./container-and-k8s-overview) | Ch1-2 | 容器技术演进、K8s 架构 |
| [核心工作负载：Pod 与副本控制](./core-workloads-pod-controllers) | Ch3-4 | Pod、ReplicaSet、DaemonSet、Job |
| [网络与服务发现](./networking-and-services) | Ch5 | Service、Ingress、DNS |
| [持久化存储](./persistent-storage) | Ch6 | Volumes、PV/PVC、StorageClass |
| [配置、密钥与应用部署](./config-secrets-deployments) | Ch7-9 | ConfigMap、Secret、Deployment |
| [有状态应用与内部架构](./statefulsets-and-internals) | Ch10-11 | StatefulSet、etcd、Scheduler |
| [安全体系](./security) | Ch12-13 | RBAC、SecurityContext、NetworkPolicy |
| 资源管理与调度 | Ch14-16 | requests/limits、HPA、taints/affinity |
| [生产最佳实践与扩展](./best-practices-and-extensions) | Ch17-18 | 生命周期、CRD、Service Catalog |

---

## 阅读方式说明

每篇笔记并非章节摘要，而是以**工程问题**为线索重新组织书中内容：先建立问题背景，再展开 K8s 的设计思路，最后落到实际使用场景。配套的交互式可视化帮助建立概念之间的关联，补充文字难以呈现的结构信息。
