---
date: 2026-04-10
title: "单节点模式：Sidecar、Ambassador 与 Adapter"
description: 《Designing Distributed Systems》Part II 阅读笔记 — 单机容器编排的三种核心设计模式
tags: [distributed-systems, sidecar, ambassador, adapter, kubernetes, containers]
---

# 单节点模式：Sidecar、Ambassador 与 Adapter

> Part II 覆盖 Chapter 3-5。这三种模式都发生在**单个节点（Pod）内部**，通过多容器组合实现关注点分离。它们是分布式系统设计模式中最基础的构建块。

## 为什么要在单机上拆分多个容器

在讨论具体模式之前，先回答一个根本问题：既然都在同一台机器上，为什么不把所有逻辑放在一个容器里？

Burns 给出三个理由：

1. **资源隔离** — 用户请求容器和后台配置加载容器可以有不同的资源限制和优先级
2. **团队扩展** — 理想团队规模是 6-8 人，拆分为小而专注的容器让不同团队可以独立拥有和部署
3. **关注点分离** — 一个 Git 同步容器可以被 PHP、Python、Node.js 等不同语言的服务复用；如果把它和运行时绑定在一起，复用就不可能了

在 Kubernetes 中，这种紧密耦合的容器组叫做 **Pod**。同一个 Pod 内的容器共享网络命名空间、文件系统卷和内存。

<HtmlVisualization src="/book-notes/designing-distributed-systems/visualizations/single-node-patterns.html" height="520px" title="三种单节点模式对比" />

## Sidecar 模式

**核心思想**：在主应用容器旁边放一个辅助容器，**扩展或增强**主容器的功能，而无需修改主容器。

### 经典案例：为遗留服务添加 HTTPS

一个老旧的 HTTP 服务，公司新规要求所有服务必须支持 HTTPS。源代码已经无法编译（构建系统过时了）。解决方案：

1. 将遗留服务配置为只监听 `localhost:80`
2. 旁边放一个 **nginx Sidecar**，监听外部 HTTPS 请求，解密后转发给 `localhost:80`
3. 因为两个容器共享网络命名空间，localhost 通信是安全的

```
外部请求 → [nginx Sidecar (HTTPS)] → localhost → [Legacy HTTP Service]
                Pod 内部，共享网络命名空间
```

### 动态配置同步

另一个常见用法：Sidecar 负责从云端配置服务拉取配置，写入共享文件系统，然后通知主应用重新加载。

1. Config Manager Sidecar 定时从 API 拉取配置
2. 检测到变化后更新共享目录中的配置文件
3. 向主应用发送信号（SIGHUP）或直接 kill 让编排器重启

### 构建简单的 PaaS

一个最简 PaaS 由两个容器组成：
- **主容器**：Node.js 服务器（使用 nodemon 监听文件变化）
- **Sidecar**：Git 同步容器（循环执行 `git pull`）

推送代码到 Git → Sidecar 拉取新代码 → 文件变化触发 Node.js 热重载。整个过程不需要 CI/CD 管道。

### 设计可复用 Sidecar 的三原则

| 原则 | 说明 |
|------|------|
| **参数化** | 通过环境变量暴露配置（如 `PORT`、`CERTIFICATE_PATH`），不硬编码 |
| **定义 API** | 容器与外界的所有交互（环境变量、HTTP 端点、共享文件）都是 API 的一部分，需要像对待微服务 API 一样维护兼容性 |
| **文档化** | 在 Dockerfile 中使用 `EXPOSE`、`ENV`、`LABEL` 指令记录端口、参数和元数据 |

> [!tip] API 兼容性陷阱
> Burns 举了一个真实案例：一个 Sidecar 的 `UPDATE_FREQUENCY` 参数最初接收秒数（如 `10`），后来改为接收带单位的字符串（如 `10s`、`5m`）。旧的纯数字值虽然不报错，但被错误地解析为毫秒。这种"看起来没破坏但实际破坏了"的 API 变更是最危险的。

## Ambassador 模式

**核心思想**：在主应用容器旁边放一个代理容器，**中介和简化**主容器与外部服务之间的通信。

与 Sidecar 的区别：Sidecar 增强主容器自身的能力，Ambassador 代理主容器与外部世界的交互。

### 案例一：分片代理（Sharded Redis）

应用需要访问一个分片的 Redis 集群，但应用代码只会连接单个 Redis 实例。解决方案：

1. 用 Kubernetes StatefulSet 部署 3 个 Redis 分片（`sharded-redis-0/1/2`）
2. 部署一个 **twemproxy** Ambassador 容器，监听 `localhost:6379`
3. 应用容器连接 `localhost:6379`，twemproxy 根据一致性哈希将请求路由到正确的分片

```yaml
# twemproxy (nutcracker) 配置
redis:
  listen: 127.0.0.1:6379
  hash: fnv1a_64
  distribution: ketama
  servers:
    - sharded-redis-0.redis:6379:1
    - sharded-redis-1.redis:6379:1
    - sharded-redis-2.redis:6379:1
```

应用代码完全不需要知道分片的存在——它只看到一个本地 Redis 端点。

### 案例二：服务发现代理（Service Broker）

在多云环境中，同一个应用可能需要连接公有云的 MySQL SaaS 或私有云的自建 MySQL。Ambassador 负责环境探测和服务发现：

```
应用容器 → connect localhost:3306 → [MySQL Service Broker Ambassador]
                                         ↓ 探测环境
                                    公有云 → 连接 SaaS MySQL
                                    私有云 → 发现并连接本地 MySQL
```

### 案例三：请求分流实验（10% Experiments）

用 nginx Ambassador 将 10% 的流量导向实验版本：

```nginx
upstream backend {
    ip_hash;           # 保证同一用户始终访问同一版本
    server web weight=9;
    server experiment;
}
```

这里使用 `ip_hash` 确保用户不会在生产版和实验版之间来回切换。

## Adapter 模式

**核心思想**：在主应用容器旁边放一个适配容器，将主容器的**异构接口标准化**为统一格式，供外部系统消费。

与前两种模式的方向相反：
- **Sidecar**：外界 → 主容器（增强输入处理能力）
- **Ambassador**：主容器 → 外界（代理输出请求）
- **Adapter**：主容器 → Adapter → 外界（标准化输出格式）

### 案例一：监控适配（Prometheus Exporter）

Redis 不原生支持 Prometheus 的 `/metrics` 接口。解决方案：在 Redis 容器旁边放一个 `redis_exporter` Adapter。

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - image: redis
    name: redis
  # Adapter：将 Redis 指标转换为 Prometheus 格式
  - image: oliver006/redis_exporter
    name: adapter
```

Prometheus 只需要知道统一的 `/metrics` 端点，不用关心底层是 Redis、MySQL 还是自定义服务。

### 案例二：日志标准化（fluentd）

不同应用有不同的日志格式（syslog、JSON、纯文本）。fluentd Adapter 将它们转换为统一的结构化格式。因为 Pod 内容器共享网络命名空间，配置只需指向 `localhost`。

### 案例三：健康检查适配（MySQL Health Monitor）

需要对 MySQL 做深度健康检查（执行一个代表性查询），但不想修改 MySQL 官方镜像。用一个 Go 写的小型 Adapter 容器：

```go
http.HandleFunc("", func(res http.ResponseWriter, req *http.Request) {
    _, err := db.Exec(*query)
    if err != nil {
        res.WriteHeader(http.StatusInternalServerError)
        return
    }
    res.WriteHeader(http.StatusOK)
})
```

Kubernetes 的 liveness/readiness 探针指向这个 Adapter 的 HTTP 端点。

> [!tip] 为什么不直接修改应用容器？
> Burns 指出：在很多情况下你在使用第三方容器镜像，fork 并维护一个修改版的成本远高于添加一个 Adapter 容器。而且 Adapter 本身是可复用的——同一个 MySQL 健康检查 Adapter 可以用于所有 MySQL 实例。

## 三种模式的统一视角

| 维度 | Sidecar | Ambassador | Adapter |
|------|---------|------------|---------|
| **数据流方向** | 增强主容器 | 代理出站请求 | 标准化出站接口 |
| **对主容器的感知** | 通常感知（共享文件/网络） | 主容器不感知分片/路由逻辑 | 主容器不感知外部格式要求 |
| **典型工具** | nginx (SSL)、git-sync | twemproxy、envoy | Prometheus exporter、fluentd |
| **OOP 类比** | Decorator 模式 | Proxy 模式 | Adapter 模式（同名！） |
| **复用性** | 高 — 跨应用通用 | 高 — 协议级通用 | 高 — 跨应用标准化 |

三种模式的共同主题是 **关注点分离**：主容器只做自己的核心业务逻辑，横切关注点（安全、路由、监控、日志）交给专门的辅助容器。这种设计使得每个容器可以被独立测试、独立更新、独立复用。
