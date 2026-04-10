---
date: "2026-04-10"
title: "有状态应用与 Kubernetes 内部架构"
description: "StatefulSet 如何给每个 Pod 稳定身份，以及 etcd、Scheduler、Controller Manager 如何协作驱动集群状态"
tags: [kubernetes, statefulset, etcd, scheduler, controller-manager, internals]
---

# 有状态应用与 Kubernetes 内部架构

> 本文对应原书 **第 10-11 章**，覆盖：StatefulSets: deploying replicated stateful applications、Understanding Kubernetes internals

---

## Ch10：StatefulSet — 给每个 Pod 一个稳定的"身份证"

### 1. 有状态应用的挑战：ReplicaSet 为什么不够用

ReplicaSet 的设计哲学是"牲畜而非宠物"（cattle, not pets）：Pod 之间完全等价，可以随时替换，新 Pod 被分配随机名称和随机 IP。这对无状态 Web 服务完美适用，但对有状态应用来说会带来两个根本性问题：

**问题一：Pod 漂移后身份丢失。** 数据库集群、分布式存储（如 Cassandra、Zookeeper、Kafka）的每个节点都需要固定地址，让其他节点能找到它。当 ReplicaSet 重建 Pod 时，新 Pod 获得全新的 hostname 和 IP，集群配置随即失效，所有成员的配置文件都需要重写。

**问题二：共享存储无法隔离。** 如果 ReplicaSet 引用一个 PVC，所有副本 Pod 会挂载同一个 PersistentVolume（图 10.1 所示）。分布式数据库要求每个节点拥有独立存储，ReplicaSet 无法做到这一点。

尝试用"每个 Pod 一个 ReplicaSet"或"在同一 PV 内建立独立子目录"也都行不通：前者让扩缩容变得极为繁琐，后者依赖应用自身的协调机制，引入更多复杂度。

### 2. StatefulSet 的三个核心保证

Kubernetes 引入 StatefulSet 来专门处理这类"有身份"的应用。StatefulSet 为其管理的 Pod 提供三个关键保证：

#### 保证一：稳定的网络身份

StatefulSet 给每个 Pod 分配一个基于有序索引的名称（0-based），而不是随机字符串。一个名为 `kubia` 的 StatefulSet 创建的 Pod 分别叫 `kubia-0`、`kubia-1`、`kubia-2`……

这与 ReplicaSet 形成鲜明对比（如图 10.5 所示）：

![StatefulSet vs ReplicaSet 命名对比](/book-notes/kubernetes-in-action/images/fig10.5-statefulset-identity.png)

当一个 StatefulSet 的 Pod 消失后（节点宕机或被手动删除），StatefulSet 会在其他节点上重建**同名的** Pod。新 Pod 获得完全相同的名称、hostname，以及（通过 Headless Service）相同的 DNS 记录。这正是有状态应用需要的"同一个人换了身体"语义（图 10.6 所示）：

![StatefulSet vs ReplicaSet Pod 替换行为](/book-notes/kubernetes-in-action/images/fig10.6-statefulset-replacement.png)

#### 保证二：稳定的专用存储

StatefulSet 清单中可以定义 `volumeClaimTemplates`。当 StatefulSet 创建每个 Pod 时，它同时按模板创建一个独立的 PVC，并将该 PVC 绑定到该 Pod（图 10.8 所示）：

![StatefulSet volumeClaimTemplate 为每个 Pod 创建独立 PVC](/book-notes/kubernetes-in-action/images/fig10.8-statefulset-pvc-template.png)

关键行为：**缩容时只删除 Pod，PVC 保留**。这是刻意的设计——防止一条误操作命令就抹去宝贵的有状态数据。等到重新扩容时，新建的同名 Pod 会自动重新挂载之前的 PVC，数据完好无损。

一个典型的 StatefulSet 清单如下：

```yaml
apiVersion: apps/v1beta1
kind: StatefulSet
metadata:
  name: kubia
spec:
  serviceName: kubia          # 指向 Headless Service
  replicas: 2
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: luksa/kubia-pet
        volumeMounts:
        - name: data
          mountPath: /var/data
  volumeClaimTemplates:       # 为每个 Pod 生成独立 PVC
  - metadata:
      name: data
    spec:
      resources:
        requests:
          storage: 1Mi
      accessModes:
      - ReadWriteOnce
```

#### 保证三：有序的部署和扩缩容

StatefulSet 严格按顺序创建 Pod：`kubia-0` 完全就绪之后，才开始创建 `kubia-1`。这避免了分布式应用在多节点同时上线时的竞争条件。

缩容同样有序，**始终先删除序号最大的 Pod**（图 10.7）。若任意 Pod 处于不健康状态，缩容操作会被暂停——因为同时失去两个节点可能导致数据丢失。

#### At-Most-One 语义

StatefulSet 有一个严格的安全保证：**任何时刻，相同身份的 Pod 最多只有一个在运行**。这意味着当节点网络分区、控制平面无法确认某个 Pod 是否还活着时，StatefulSet 不会贸然创建替代 Pod。要强制重建，管理员必须显式地执行 `kubectl delete pod kubia-0 --force --grace-period 0`，并且清楚地知道原 Pod 确实已停止。

### 3. Headless Service + DNS 对等发现

StatefulSet 必须配合一个 **Headless Service**（`clusterIP: None`）。这个 Service 不提供虚拟 IP，而是直接在 DNS 中为每个 Pod 注册 A 记录。

每个 Pod 获得如下格式的全限定域名（FQDN）：

```
<pod-name>.<service-name>.<namespace>.svc.cluster.local
```

例如，`kubia-0` 在 `default` 命名空间中可通过 `kubia-0.kubia.default.svc.cluster.local` 访问。

更强大的是 **DNS SRV 记录**：对 Headless Service 执行 SRV 查询，可以一次性获取所有 Pod 的地址列表：

```
$ dig SRV kubia.default.svc.cluster.local
;; ANSWER SECTION:
kubia.default.svc.cluster.local. 30 IN SRV 10 33 0 kubia-0.kubia.default.svc.cluster.local.
kubia.default.svc.cluster.local. 30 IN SRV 10 33 0 kubia-1.kubia.default.svc.cluster.local.
```

这样，应用内部可以通过一次 SRV 查询发现所有集群成员，无需依赖 Kubernetes API，保持了 Kubernetes 无感知的设计原则。

### 4. AI 推理场景的应用

StatefulSet 在 AI/ML 工程中有大量实际用例：

**分布式训练的 Parameter Server 模式**。PS-Worker 架构中，Parameter Server 节点（PS-0、PS-1）需要稳定的 DNS 地址供众多 Worker 节点连接。用 StatefulSet 部署 PS 节点，Worker 可以硬编码 `ps-0.ps-service.training.svc.cluster.local` 作为连接地址，即使 PS-0 所在节点崩溃重建，Worker 重连时地址依然有效。

**向量数据库集群**。Qdrant、Milvus、Weaviate 这类向量数据库的分布式部署全部依赖 StatefulSet：
- 每个节点维护自己的向量分片（独立 PVC）
- 节点间通过稳定 DNS 互相发现、同步状态
- 扩容时按序加入集群，确保数据重平衡过程可控

**KV Cache 服务**。在大模型推理链路中，用 StatefulSet 部署的 KV Cache 节点（如 Mooncake、InfiniStore）通过稳定身份实现请求路由——特定的 KV Cache 分片始终映射到固定的 Pod，避免缓存失效。

---

## Ch11：Kubernetes 内部架构 — 控制平面的协作机制

### 5. etcd：集群的唯一真相（SSOT）

etcd 是整个 Kubernetes 集群的持久化后端，一个分布式的键值存储。所有 Kubernetes 资源对象——Pod、Service、Deployment、ConfigMap——都以 JSON 格式存储在 etcd 的 `/registry/` 路径下。

**为什么其他组件不能直接写 etcd？** Kubernetes 强制要求所有组件通过 API Server 读写 etcd，原因有三：
1. API Server 统一实施认证、授权、准入控制
2. API Server 内置乐观并发锁（optimistic locking），通过 `resourceVersion` 字段防止并发写冲突
3. 隐藏存储实现细节，未来可以替换 etcd

**Raft 共识协议。** etcd 通常以 3/5/7 节点集群部署（奇数节点），使用 Raft 协议保证一致性：写操作必须获得多数节点（quorum）确认才能提交。如果集群发生网络分区，只有持有多数节点的分区可以继续接受写请求；少数节点分区进入只读状态，防止"脑裂"（图 11.2 所示）。

### 6. API Server：集群的唯一入口

API Server 是 Kubernetes 控制平面的核心网关，提供 RESTful CRUD 接口。所有对集群状态的操作都经过 API Server，流程如下（图 11.3）：

```
HTTP 请求 → 认证插件（Authentication）
         → 授权插件（Authorization，如 RBAC）
         → 准入控制插件（Admission Control）
         → 资源校验
         → 写入 etcd
         → 返回响应
```

**Watch 机制是一切事件驱动的基础。** API Server 支持客户端建立长连接并 Watch 特定资源类型。任何资源变更（增/改/删）发生时，API Server 将变更事件推送给所有正在 Watch 的客户端。Scheduler、各 Controller、Kubelet 都通过这个机制实时感知集群变化，无需轮询（图 11.4）。

这也是为什么 API Server 是整个系统最关键的单点：它是 etcd 的唯一客户端，也是所有其他组件的"神经中枢"。

### 7. Scheduler：两阶段调度

Scheduler 监听（Watch）尚未绑定节点的 Pod，对每个待调度的 Pod 执行两阶段算法（图 11.5）：

**第一阶段 — Filtering（过滤）**：筛除不满足 Pod 需求的节点。检查项包括：
- 节点资源是否满足 Pod 的 CPU/内存 requests
- 节点是否符合 Pod 的 nodeSelector 标签
- Pod 需要的 Volume 类型节点是否支持
- 节点是否存在 Pod 不容忍（tolerate）的污点（taint）
- Pod 的亲和/反亲和规则是否满足

**第二阶段 — Scoring（打分）**：对通过过滤的节点打分，选出最优节点。评分维度包括当前负载、数据局部性（镜像是否已预拉取）、Pod 扩散分布等。

Scheduler 通过 API Server 将选定的节点名写入 Pod 的 `spec.nodeName` 字段，然后 Kubelet 感知到这个变化并启动容器。Scheduler 本身不和 Kubelet 直接通信。

### 8. Controller Manager：控制循环大集合

Controller Manager 是一个进程，内部运行着十多个控制器，每个控制器负责一类资源的 Reconciliation Loop（协调循环）：

| 控制器 | 职责 |
|--------|------|
| ReplicaSet Controller | 确保 Pod 副本数与期望一致 |
| Deployment Controller | 管理滚动更新，操作 ReplicaSet |
| StatefulSet Controller | 管理有状态 Pod 和 PVC 的生命周期 |
| Node Controller | 监控节点健康，驱逐失联节点上的 Pod |
| Endpoints Controller | 维护 Service 的 Endpoints 列表（IP:Port） |
| Namespace Controller | 删除命名空间时清理其下所有资源 |
| PersistentVolume Controller | 将 PVC 与合适的 PV 绑定 |

**控制循环的通用模式**：

```
1. Watch API Server → 感知期望状态变化（spec）
2. 读取实际状态（status）
3. 计算差异（desired - actual）
4. 通过 API Server 采取动作，消除差异
5. 更新 status 字段，记录实际状态
```

所有控制器只通过 API Server 操作，彼此之间完全不直接通信。API Server 是它们唯一的协作媒介。

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/k8s-control-loop.html" height="480px" title="K8s 控制循环演示" />

### 9. Kubelet：节点的守护进程

Kubelet 是唯一运行在 Worker Node 上（而非 Master Node）的控制平面组件，也是整个系统中唯一直接管理容器的组件。它的职责包括：

1. **节点注册**：启动时在 API Server 创建 Node 资源，上报节点容量信息
2. **Pod 启动**：Watch API Server，当新 Pod 被调度到本节点时，调用容器运行时（Docker/containerd）启动容器
3. **健康监控**：运行 liveness/readiness probe，失败时重启容器
4. **状态上报**：定期向 API Server 汇报容器和节点的实时状态
5. **优雅停止**：Pod 被删除时执行 preStop hook，等待 gracePeriod 后强制终止容器

Kubelet 有一个特别的能力：它也可以从本地清单目录（manifest directory）启动 Pod，无需 API Server。Kubernetes 控制平面组件（API Server、Scheduler 等）本身就是以这种"静态 Pod"方式在 Master 上运行的，由 Master 上的 Kubelet 管理。

**运行 Pod 的真相**。当你创建一个"单容器 Pod"时，实际上在节点上会启动两个容器：应用容器 + 一个名为 `pause` 的基础设施容器（图 11.13）。`pause` 容器持有 Pod 的 Linux 网络命名空间和 IPC 命名空间。应用容器共享这些命名空间，这样即使应用容器崩溃重启，它依然能回到相同的网络环境中。

### 10. 一个 Deployment 创建的完整事件链

理解了各组件的职责后，让我们串联起来，看看 `kubectl apply -f deployment.yaml` 之后究竟发生了什么（图 11.12）：

**事件链（全程基于 Watch 机制驱动）**：

```
Step 1: kubectl apply
  → kubectl POST /apis/apps/v1/namespaces/default/deployments
  → API Server 认证 → 授权 → 准入控制 → 写入 etcd
  → 返回 200 OK 给 kubectl

Step 2: Deployment Controller 感知
  → 收到 Watch 通知："有新 Deployment 创建"
  → 读取 Deployment spec（replicas=3）
  → 创建对应的 ReplicaSet 资源，POST 到 API Server

Step 3: ReplicaSet Controller 感知
  → 收到 Watch 通知："有新 ReplicaSet 创建"
  → 检查当前 Pod 数量（0） vs 期望数量（3）
  → 批量创建 3 个 Pod 资源（仅创建 API 对象，无节点绑定）

Step 4: Scheduler 感知
  → 收到 Watch 通知："有 nodeName 为空的新 Pod"
  → 对每个 Pod 执行 Filtering + Scoring
  → PATCH pod.spec.nodeName = "node-X"（写回 API Server）

Step 5: Kubelet 感知
  → Node-X 上的 Kubelet 收到 Watch 通知："有新 Pod 调度到本节点"
  → 拉取镜像（如未缓存）
  → 调用 containerd 创建并启动容器
  → 更新 Pod status 为 Running，上报 API Server
```

整个过程中，没有任何组件直接调用另一个组件。所有协作都通过 API Server 的 Watch 机制串联，形成一个完全去中心化的事件驱动系统。

---

## 总结

| 组件 | 类型 | 核心职责 | 关键特性 |
|------|------|---------|---------|
| **StatefulSet** | 工作负载资源 | 管理有状态 Pod | 稳定身份、独立 PVC、有序操作、At-Most-One |
| **etcd** | 存储 | 持久化所有集群状态 | Raft 一致性、唯一真相来源、只能通过 API Server 访问 |
| **API Server** | 控制平面 | 集群网关、状态存取 | 认证/授权/准入控制、Watch 机制、乐观并发锁 |
| **Scheduler** | 控制平面 | Pod 节点绑定 | 两阶段：Filter + Score、只写 nodeName |
| **Controller Manager** | 控制平面 | 运行所有 Reconciliation Controller | 控制循环模式、只通过 API Server 操作 |
| **Kubelet** | 工作节点 | 启动和管理容器 | Watch Pod 变化、调用容器运行时、上报状态 |
| **kube-proxy** | 工作节点 | Service 流量路由 | 配置 iptables 规则、Watch Services 和 Endpoints |

**核心思想**：Kubernetes 是一个声明式的、事件驱动的控制系统。用户描述"期望状态"，各 Controller 的 Reconciliation Loop 不断观察实际状态与期望状态的差异，并采取行动消除差异。这套模式——Observe（观察）→ Diff（比较）→ Act（行动）——是理解 Kubernetes 所有行为的钥匙。
