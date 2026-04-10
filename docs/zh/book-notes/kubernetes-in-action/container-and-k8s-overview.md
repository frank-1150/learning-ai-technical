---
date: 2026-04-10
title: "容器化与 Kubernetes 全局观"
description: "从单体到微服务的架构演进，理解容器技术与 Kubernetes 解决的本质问题"
tags: [kubernetes, docker, containers, microservices]
---

# 容器化与 Kubernetes 全局观

> 本文对应原书 **第 1-2 章**，覆盖：Introducing Kubernetes、First steps with Docker and Kubernetes

---

## 1. 为什么需要容器化

### 单体应用的时代

十年前，大多数软件是大型单体应用——要么以单一进程运行，要么以少量进程分布在几台服务器上。这类系统发布周期漫长，通常由开发团队打包好整个系统，交给运维团队部署、维护。当硬件出现故障时，运维人员手动将应用迁移到其他健康的服务器。

单体应用的扩容方式有两种：
- **垂直扩容（Scale Up）**：加更多 CPU、内存、磁盘，成本高、有上限
- **水平扩容（Scale Out）**：增加机器跑多个副本，便宜，但要求应用本身支持

> [!note]
> 单体应用的致命弱点：应用内部任何一个模块不支持水平扩容，整个系统就无法横向扩展。关系型数据库就是经典案例。

### 微服务时代的新问题

现代应用被拆分为**微服务（Microservices）**——一组独立部署、独立运行的小型服务，通过 HTTP REST 或 AMQP 等协议相互通信。这解决了单体的可维护性和独立扩容问题，但引入了新的工程挑战：

**挑战一：部署复杂度爆炸**

当系统有几十个微服务时，每个服务的部署位置、配置、版本依赖都不同。手工管理这些组合几乎不可能——服务数量增加是线性的，但服务间依赖和部署组合是指数级增加的。

**挑战二：依赖冲突（Dependency Hell）**

不同团队开发的微服务可能依赖同一个库的不同版本。在同一台服务器上运行多个应用时，Library A v1.0 和 Library A v2.2 无法共存，导致所谓的"依赖地狱"。

原书 Figure 1.3 直观展示了这个问题：右侧四个应用运行在同一台服务器上，依赖 Library A 的 v1.0、v2.2 两个不兼容版本。

**挑战三：环境不一致**

开发机、测试环境、生产环境在操作系统版本、已安装库、网络配置上都有差异。"在我机器上好好的"是工程师最常听到的抱怨之一。

这三个问题，正是容器技术要解决的核心。

---

## 2. 容器 vs 虚拟机：隔离原理对比

### 虚拟机：重量级隔离

虚拟机（VM）通过 Hypervisor 将物理硬件分割成多组虚拟资源，每个 VM 运行独立的 Guest OS。VM 内的应用调用系统调用时，路径是：应用 → Guest OS Kernel → Hypervisor → 物理 CPU。

这提供了**完整的隔离**，但代价昂贵：
- 每个 VM 都要运行完整的操作系统（几 GB 磁盘、几百 MB 内存）
- VM 启动需要分钟级时间（需要引导操作系统）
- 多应用必须塞进同一个 VM，以避免资源浪费

### 容器：轻量级隔离

容器共享宿主机的操作系统内核，不需要独立的 Guest OS。一个容器本质上是宿主机上一个特殊的**隔离进程**——它只看到宿主机内核为它划定的那部分视图。

![容器 vs 虚拟机对比图](/book-notes/kubernetes-in-action/images/fig-1-4-vm-vs-container.png)

*Figure 1.4（原书）：三台 VM 运行三个应用 vs. 容器隔离同等数量的应用。容器完全不需要 Guest OS 层。*

容器隔离依赖 Linux 内核的两个核心机制：

**Linux Namespaces — 视图隔离**

Namespace 让每个容器只能看到系统的一个子集。当前有 6 种 Namespace：

| Namespace | 隔离内容 |
|-----------|---------|
| Mount (mnt) | 文件系统挂载点 |
| Process ID (pid) | 进程 ID 空间（容器内 PID 1 是容器主进程） |
| Network (net) | 网络接口、路由表、端口 |
| Inter-process communication (ipc) | 共享内存、信号量 |
| UTS | 主机名和域名（容器看到的 hostname） |
| User ID (user) | 用户 ID 和组 ID 映射 |

**Linux Control Groups (cgroups) — 资源限制**

Namespace 解决"看什么"的问题，cgroups 解决"能用多少"的问题。cgroups 是 Linux 内核特性，用于限制进程（或进程组）可消耗的 CPU、内存、网络带宽等资源。这样一个容器无法抢占其他容器预留的资源，等效于让每个进程运行在独立机器上。

> [!tip]
> **关键理解**：Docker 本身不提供隔离。真正的隔离由 Linux 内核的 Namespace 和 cgroups 完成。Docker 只是让这些内核特性易于使用，并把应用打包成可移植的镜像。

### 性能对比

| 维度 | 虚拟机 | 容器 |
|------|--------|------|
| 启动时间 | 分钟级（需引导 OS） | 秒级甚至毫秒级 |
| 磁盘占用 | GB 级（完整 OS） | MB 级（仅应用层） |
| 内存开销 | 数百 MB（OS 自身消耗） | 几乎为零（共享内核） |
| 隔离级别 | 完整（各自独立内核） | 进程级（共享内核，有安全风险） |
| 适合场景 | 不同 OS、强安全隔离 | 同 OS、高密度部署 |

---

## 3. Docker 核心概念

### Docker 是什么

Docker 是容器技术的第一个让容器跨机器真正可移植的平台。它把应用及其整个运行环境（库、系统文件、配置）打包进一个可移植的镜像，使得这个镜像可以在任何安装了 Docker 的机器上运行，并看到完全一致的文件系统。

### 三个核心概念

**镜像（Image）**

Docker 镜像是包含应用及其运行环境的打包单元。它采用**分层（Layers）**结构：每个 Dockerfile 指令创建一个新层，层之间共享。不同镜像可以引用相同的底层，节省存储和传输带宽。

```
kubia:latest image          other:latest image
┌─────────────────────┐    ┌─────────────────────┐
│  ADD app.js/app.js  │    │  CMD node app.js     │
│  CMD node           │    │  ADD app.js/app.js   │
├─────────────────────┤    ├─────────────────────┤
│  (node:0.12 layers) │    │  (node:0.12 layers) │  ← 共享层
└─────────────────────┘    └─────────────────────┘
```

**仓库（Registry）**

Docker Registry 是存储和分发镜像的服务。最广为人知的是 Docker Hub（hub.docker.com），也有 Quay.io、Google Container Registry 等。开发者把镜像 push 到仓库，其他机器从仓库 pull。

**容器（Container）**

容器是镜像运行时的实例——一个从镜像创建的隔离进程，资源受限，有自己的 PID 空间、网络接口和文件系统视图。

### Dockerfile 示例

原书用一个简单的 Node.js 应用演示完整流程：

```dockerfile
FROM node:7
ADD app.js /app.js
ENTRYPOINT ["node", "app.js"]
```

配套的 `app.js`（Node.js HTTP 服务器，返回容器的 hostname）：

```javascript
const http = require('http');
const os = require('os');

var handler = function(request, response) {
  response.writeHead(200);
  response.end("You've hit " + os.hostname() + "\n");
};

var www = http.createServer(handler);
www.listen(8080);
```

> [!note]
> 当应用被扩展到多个实例时，每个容器返回的 hostname 不同，正好验证了负载均衡在多个 Pod 之间分发请求。

### Docker 工作流

```bash
# 1. 构建镜像
docker build -t kubia .

# 2. 本地运行测试
docker run --name kubia-container -p 8080:8080 -d kubia
curl localhost:8080  # "You've hit <container-id>"

# 3. 推送到仓库
docker tag kubia yourid/kubia
docker push yourid/kubia

# 4. 在任何机器上运行
docker run -p 8080:8080 -d yourid/kubia
```

进入运行中的容器查看内部状态：

```bash
docker exec -it kubia-container bash
# 容器内只能看到自己的进程
root@<id>:/# ps aux
USER  PID  COMMAND
root    1  node app.js   # PID 1 是容器主进程
```

---

## 4. Kubernetes 是什么

### 从 Google Borg 到开源

Google 长期是世界上运行服务器规模最大的公司之一，数十万台服务器上部署了数千个软件组件。手工管理这个规模在经济上不可行，促使 Google 内部开发了 **Borg** 系统（以及后继的 **Omega**），实现了应用的自动调度和管理，大幅提升硬件利用率。

2014 年，Google 将积累了十年的工程经验开源，发布了 **Kubernetes**（希腊语，意为"舵手"或"飞行员"）。

### Kubernetes 解决什么问题

Kubernetes 把整个数据中心的硬件资源抽象为**一台超大计算机**，让开发者不再需要知道应用运行在哪台服务器上。你只需要告诉 K8s"我要运行什么、需要多少资源"，K8s 负责决定放在哪、保持运行、在故障时自动恢复。

对**开发者**：
- 部署应用无需了解底层基础设施
- 不必实现服务发现、负载均衡、自动重启等功能，K8s 直接提供
- 开发和生产环境一致，减少 "在我机器上好好的" 问题

对**运维人员**：
- 从管理每个应用，转变为管理 K8s 和基础设施本身
- 服务器节点故障时，K8s 自动迁移上面的应用，无需凌晨三点被叫醒

> [!tip]
> K8s 的本质是一套**声明式系统（Declarative System）**：你描述期望的状态（desired state），K8s 持续协调实际状态（actual state）向期望状态收敛。这被称为 Reconciliation Loop，是分布式系统设计的核心模式。

---

## 5. K8s 集群架构

K8s 集群由两类节点组成：

- **Control Plane（主节点）**：控制和管理整个 K8s 系统
- **Worker Nodes（工作节点）**：实际运行容器化应用

![K8s 集群架构图](/book-notes/kubernetes-in-action/images/fig-1-9-k8s-cluster-architecture.png)

*Figure 1.9（原书）：Kubernetes 集群的组件构成。*

### Control Plane 组件

**API Server**

整个集群的入口和通信中心。kubectl、其他 Control Plane 组件、Worker Node 上的 Kubelet，所有请求都通过 API Server。它提供 RESTful API，负责认证、授权和数据验证。

**etcd**

分布式键值存储，持久化保存集群的所有配置和状态数据（Pod 定义、Service 配置、节点信息等）。整个集群的"单一事实来源（Single Source of Truth）"。etcd 是高可用设计，通常以集群模式运行多个副本。

**Scheduler**

调度器负责决定新 Pod 应该运行在哪个 Worker Node 上。调度时考虑节点的可用资源、Pod 的资源需求、亲和性/反亲和性规则等。一旦决定，将调度结果写入 API Server，由对应节点的 Kubelet 执行。

**Controller Manager**

运行多个控制器（Controller），每个控制器负责监控集群某个方面的状态并进行调整：
- **Replication Controller**：确保指定数量的 Pod 副本始终在运行
- **Node Controller**：监控节点健康，处理节点故障
- **Endpoint Controller**：维护 Service 到 Pod 的端点映射

### Worker Node 组件

**Kubelet**

Worker Node 上的"代理"，和 API Server 保持通信。负责在本节点上运行和管理 Pod，监控容器健康状态，并上报给 API Server。

**kube-proxy**

每个 Worker Node 上运行的网络代理，负责维护 iptables/ipvs 规则，实现 Service 的网络负载均衡。当请求到达 Service IP 时，kube-proxy 将流量转发到对应的 Pod。

**Container Runtime**

实际运行容器的引擎（Docker、containerd、CRI-O 等）。Kubelet 通过 CRI（Container Runtime Interface）调用它创建和管理容器。

> [!tip]
> **AI 推理服务场景**：以 LLM 推理集群为例，每个模型实例（如 vLLM 进程）运行在独立容器中。Scheduler 根据 GPU 资源需求（`nvidia.com/gpu: 1`）将 Pod 调度到有空闲 GPU 的节点上；Controller Manager 确保始终有指定数量的推理副本在运行；kube-proxy 在多个推理实例之间实现请求的负载均衡。这就是 K8s 在 AI 基础设施中的典型价值。

### 完整架构交互图

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/k8s-cluster-architecture.html" height="500px" title="Kubernetes 集群架构交互图" />

---

## 6. 在 K8s 上运行第一个应用

### Pod：K8s 的基本调度单位

K8s 不直接管理容器，而是通过 **Pod** 这个抽象来工作。Pod 是一组紧密相关的容器的集合，它们：
- 始终在同一个 Worker Node 上运行
- 共享同一组 Linux Namespace（包括 Network namespace，即共享 IP 和端口空间）
- 对外表现为一台独立的逻辑机器，有自己的 IP 和 hostname

通常一个 Pod 只包含一个容器，但有时需要将紧密协作的容器放在同一个 Pod（如 sidecar 模式）。

### 完整部署流程

![kubectl run 完整流程图](/book-notes/kubernetes-in-action/images/fig-2-6-kubectl-run-flow.png)

*Figure 2.6（原书）：从 `kubectl run` 到容器在节点上运行的完整流程。*

**步骤一：构建并推送镜像**

```bash
docker build -t kubia .
docker tag kubia yourid/kubia
docker push yourid/kubia
```

**步骤二：部署到 K8s**

```bash
kubectl run kubia \
  --image=yourid/kubia \
  --port=8080 \
  --generator=run/v1
# 输出：replicationcontroller "kubia" created
```

这个命令在后台做了几件事：
1. 向 API Server 发送 REST 请求，创建 ReplicationController
2. Scheduler 看到新的 Pod 需要调度，选择一个合适的 Worker Node
3. 目标节点的 Kubelet 被通知，指示 Docker 从镜像仓库拉取镜像
4. Docker 创建容器并运行

**步骤三：查看 Pod 状态**

```bash
kubectl get pods
# NAME          READY   STATUS    RESTARTS   AGE
# kubia-4jfyf   0/1     Pending   0          1m   ← 镜像拉取中

kubectl get pods
# NAME          READY   STATUS    RESTARTS   AGE
# kubia-4jfyf   1/1     Running   0          5m   ← 运行中
```

**步骤四：暴露服务**

Pod 的 IP 是集群内部 IP，外部无法直接访问。需要创建 Service 来暴露：

```bash
kubectl expose rc kubia \
  --type=LoadBalancer \
  --name kubia-http
# service "kubia-http" exposed

kubectl get services
# NAME         CLUSTER-IP      EXTERNAL-IP     PORT(S)          AGE
# kubia-http   10.3.246.185    104.155.74.57   8080:31348/TCP   1m

curl 104.155.74.57:8080
# You've hit kubia-4jfyf
```

**步骤五：水平扩容**

K8s 的扩容只需一条命令：

```bash
kubectl scale rc kubia --replicas=3
# replicationcontroller "kubia" scaled

kubectl get pods
# NAME          READY   STATUS    RESTARTS   AGE
# kubia-hczji   1/1     Running   0          7s
# kubia-iq9y6   0/1     Pending   0          7s
# kubia-4jfyf   1/1     Running   0          18m

# 多次访问，请求被分发到不同 Pod
curl 104.155.74.57:8080  # You've hit kubia-hczji
curl 104.155.74.57:8080  # You've hit kubia-iq9y6
curl 104.155.74.57:8080  # You've hit kubia-4jfyf
```

> [!note]
> 扩容时你不是在"命令 K8s 再加两个 Pod"，而是在"声明期望有 3 个副本"。K8s 看到当前状态是 1 个，期望是 3 个，自动创建 2 个。这就是声明式系统的核心——你只描述 **what**，K8s 决定 **how**。

### 逻辑组件关系

原书 Figure 2.7 展示了三个核心逻辑对象的关系：

```
外部请求 → Service (kubia-http)
              IP: 10.3.246.185 (内部)
              IP: 104.155.74.57 (外部)
                ↓ 负载均衡
            Pod (kubia-4jfyf)
              IP: 10.1.0.1
              Container (node app.js)
                ↑ 副本控制
         ReplicationController (kubia)
              Replicas: 3
```

- **ReplicationController**：确保始终有指定数量的 Pod 副本在运行，Pod 故障时自动创建替换
- **Service**：为一组 Pod 提供稳定的 IP 和 DNS 名，因为 Pod 的 IP 是临时的（Pod 重建后 IP 会变）
- **Pod**：实际运行容器的基本单位，是 K8s 调度的最小对象

---

## 总结：核心概念速查表

| 概念 | 定义 | 解决的问题 |
|------|------|-----------|
| **Container** | 使用 Linux Namespace + cgroups 隔离的进程 | 依赖冲突、环境不一致 |
| **Docker Image** | 应用及其完整运行环境的分层打包 | 环境差异、可移植性 |
| **Docker Registry** | 镜像的分发仓库（Docker Hub 等） | 镜像共享和分发 |
| **Pod** | K8s 最小调度单位，一组共享网络的容器 | 紧密协作的容器需要同机运行 |
| **ReplicationController** | 确保指定数量 Pod 副本持续运行 | 容器崩溃后的自动恢复 |
| **Service** | 为一组 Pod 提供稳定 IP 和负载均衡 | Pod IP 不固定导致无法稳定访问 |
| **API Server** | K8s 集群的 REST 入口和通信中心 | 统一管理所有集群操作 |
| **etcd** | 集群配置和状态的分布式持久化存储 | 集群状态的单一事实来源 |
| **Scheduler** | 决定 Pod 运行在哪个 Worker Node | 资源感知的智能调度 |
| **Controller Manager** | 运行多个控制器，持续调和期望和实际状态 | 声明式系统的执行引擎 |
| **Kubelet** | Worker Node 上的代理，管理节点上的 Pod | 在节点层面执行调度决定 |
| **kube-proxy** | 节点上的网络代理，实现 Service 网络规则 | 服务发现和负载均衡 |
| **Linux Namespace** | 隔离进程的系统视图（PID、网络、文件系统等） | 容器间的进程隔离 |
| **cgroups** | 限制进程可消耗的 CPU、内存、网络等资源 | 防止容器抢占其他容器资源 |
| **Dockerfile** | 定义镜像构建过程的指令文件 | 镜像构建的可重复性和自动化 |

> [!tip]
> **本章最重要的思维转变**：从"告诉系统在哪台机器上做什么"（命令式，Imperative），转变为"告诉系统期望的最终状态是什么"（声明式，Declarative）。这个思维模型不仅适用于 K8s，也是现代分布式系统设计的通用范式。
