---
date: 2026-04-10
title: "资源管理、弹性扩缩容与高级调度"
description: "从 CPU/内存 requests/limits 到 GPU 调度，从 HPA 弹性伸缩到节点亲和性与污点容忍——K8s 资源调度的完整体系"
tags: [kubernetes, resources, hpa, scheduling, affinity, taints, tolerations, qos, gpu]
---

# 资源管理、弹性扩缩容与高级调度

> 本文对应原书 **第 14-16 章**，覆盖：Managing pods' computational resources（Ch14）、Automatic scaling of pods and cluster nodes（Ch15）、Advanced scheduling（Ch16）

这三章是整本书对 AI 推理基础设施最关键的内容。它们回答了一个核心问题：**在多租户 Kubernetes 集群中，如何让每个 Pod 既能拿到它需要的资源，又不会掠夺其他 Pod 的份额？**

理解资源调度不是运维细节，它是 AI 平台工程师的核心能力。一个没有正确设置 requests/limits 的 LLM 推理集群，要么因为资源碎片化导致 GPU 空转浪费，要么因为内存 OOM 在高峰期集体崩溃——两种结局都会让业务直接受损。

---

## Ch14：请求与限制——资源调度的双层保障

### 1. requests 与 limits：两个不同的概念

在 Kubernetes 中，每个容器的资源声明分为两个独立字段：

- **`requests`**：容器"保证能拿到"的最低资源量。Scheduler 用它来决定 Pod 应该调度到哪个节点。
- **`limits`**：容器"最多能用"的上限。运行时（kubelet + cgroup）用它来约束容器的资源消耗。

为什么要分开这两个字段？因为它们服务于不同的目的：
- **调度时**：节点需要知道最低保障量，才能判断自己能不能容纳这个 Pod。用 limits 来调度会导致过度保守——limits 可能远高于实际需求。
- **运行时**：防止某个容器无限制扩张，挤占同节点上其他 Pod 的资源。

```yaml
# 一个 LLM 推理 Pod 的资源声明示例
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: vllm-server
    image: vllm/vllm-openai:latest
    resources:
      requests:
        cpu: "4"
        memory: "32Gi"
        nvidia.com/gpu: "1"   # 扩展资源，整数语义
      limits:
        cpu: "8"
        memory: "40Gi"
        nvidia.com/gpu: "1"   # GPU 限制必须等于 requests
```

### CPU 与内存的根本区别

CPU 和内存在超限时的行为截然不同，这个区别对 AI 工作负载尤为关键：

**CPU 是可压缩资源（compressible）**。当容器想用更多 CPU 但已超出限制，内核只是降低它的 CPU 时间片分配（throttle），进程不会死——只是变慢。这类似于限速，不是封禁。

**内存是不可压缩资源（incompressible）**。内存一旦分配给进程，在进程主动释放之前，内核无法强制回收。当容器内存超过 limits，内核 OOM Killer 会直接杀死该容器（`OOMKilled`，Exit Code 137）。如果 Pod 的 restartPolicy 是 `Always`，Kubelet 会重启容器，但反复 OOM 会触发 `CrashLoopBackOff`，重启间隔从 10s 指数增长到 300s 封顶。

**AI 推理中的意义**：一个 vLLM 推理服务在处理长上下文（128K tokens）时，KV cache 的内存占用会急剧上升。如果 limits 设置太低，高峰期的请求直接触发 OOM，服务中断。正确做法是基于实际压测数据设置 limits，并且给 KV cache 留出足够的 headroom。

### 2. Scheduler 如何利用 requests 选节点

Scheduler 在调度时只看 requests，从不看节点的实际资源使用量。这个设计看似奇怪，但有深刻的工程理由：

> **Scheduler 必须保证它做出的承诺是可信的。**

如果 Scheduler 把一个 Pod 调度到某节点，是基于该节点"当前 CPU 用量低"——那下一秒其他 Pod 突然开始用 CPU，这个保证就破裂了。基于 requests 调度，是基于"已经预留给哪些 Pod"的确定性信息，而不是瞬时状态。

调度的三个阶段：

1. **Filter（过滤）**：排除所有"节点可分配资源 < Pod 的 requests"的节点。`Allocatable` 资源 = 节点总量 - 系统/K8s 组件预留量。
2. **Score（打分）**：对剩余节点打分。`LeastRequestedPriority` 偏好空闲资源多的节点（分散负载）；`MostRequestedPriority` 偏好已使用资源多的节点（装箱，降低节点数量，节省云成本）。
3. **Bind（绑定）**：将 Pod 绑定到得分最高的节点。

**LimitRange：为命名空间设置默认值和边界**

如果团队成员创建 Pod 时忘记设置 requests/limits，Pod 以 BestEffort 模式运行——这在生产集群很危险。LimitRange 解决这个问题：

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: team-ai-limits
  namespace: ai-inference
spec:
  limits:
  - type: Container
    defaultRequest:
      cpu: 500m
      memory: 512Mi
    default:
      cpu: "2"
      memory: 4Gi
    max:
      cpu: "16"
      memory: 64Gi
    maxLimitRequestRatio:
      cpu: 4        # limits 最多是 requests 的 4 倍
```

LimitRange 由 `LimitRanger` Admission Controller 在 Pod 创建时执行——不符合 min/max 的 Pod 会被直接拒绝，不符合 defaultRequest/default 的容器会自动注入默认值。

### 3. QoS 类别：内存压力下的优先级保障

当节点内存严重不足时，Kubernetes 需要决定杀死哪些容器。这个决策不是随机的，而是基于 Pod 的 **QoS（Quality of Service）类别**。

QoS 类别不是显式设置的，而是由 requests 和 limits 的关系自动推导：

| QoS 类别 | 判定条件 | 驱逐优先级 |
|---------|---------|-----------|
| **BestEffort** | 所有容器均未设置 requests 和 limits | 最低（最先被杀） |
| **Burstable** | 至少一个容器的 requests < limits，或只设了其中之一 | 中等 |
| **Guaranteed** | 所有容器的 requests == limits（CPU 和内存均需满足） | 最高（最后被杀） |

多容器 Pod 的 QoS 取最低类别。只要有一个容器是 BestEffort，整个 Pod 就是 BestEffort。

**OOM Score 机制**：同一 QoS 类别内的多个容器，按 OOM Score 排序——OOM Score 越高越先被杀。BestEffort Pod 的 OOM Score 默认最高（1000），Guaranteed Pod 最低（-998 或更低）。同类 Burstable Pod 之间，**实际使用量/请求量**比值越高，OOM Score 越高，越先被杀。

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/qos-eviction.html" height="480px" title="QoS 驱逐优先级演示" />

**在 AI 平台中的实践**：
- **在线推理服务**（vLLM、TGI）：必须是 Guaranteed QoS。`requests == limits`，绝对不能被 OOM 驱逐。
- **离线批处理任务**（数据预处理、模型评估）：Burstable 可接受，但要设置合理的内存 requests，防止被过早驱逐。
- **开发调试 Pod**：BestEffort 可以，节省资源，但随时可能被驱逐——不要在上面跑重要任务。

### 4. ResourceQuota：限制命名空间总资源

LimitRange 管理单个 Pod 的边界，ResourceQuota 管理整个命名空间的总量上限：

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-quota
  namespace: team-a
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 200Gi
    limits.cpu: "40"
    limits.memory: 400Gi
    requests.nvidia.com/gpu: "4"    # GPU 配额
    pods: "50"
    persistentvolumeclaims: "20"
```

在多团队共享 GPU 集群中，ResourceQuota 是防止某个团队"跑满全部 GPU"的关键机制。配合 LimitRange（设置单个 Pod 的 GPU 上限），可以实现细粒度的资源治理。

---

## Ch15：弹性扩缩容——让集群自动应对负载变化

### 5. HPA：水平 Pod 自动扩缩容

HorizontalPodAutoscaler（HPA）是 Kubernetes 中最常用的自动扩缩容机制。它通过调整 Deployment/ReplicaSet 的副本数，让服务能自动应对负载波动。

**工作流程**：

1. **采集指标**：cAdvisor（运行在每个 kubelet 中）收集 Pod 的资源使用数据，汇总到 Metrics Server（或 Prometheus）。
2. **计算期望副本数**：HPA Controller 周期性（默认 15s）查询指标，按公式计算：

   ```
   desiredReplicas = ceil(currentMetric / targetMetric * currentReplicas)
   ```

   例如：3 个 Pod，当前平均 CPU 利用率 108%，目标 30%：
   `ceil(108 / 30 * 3) = ceil(10.8) = 4` — 扩容到 4 个副本。

3. **更新副本数**：HPA 修改目标资源的 `scale` 子资源（replicas 字段），Deployment Controller 负责实际创建/删除 Pod。

**HPA YAML 示例**：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods           # 自定义 Pod 指标
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "20"  # 每个 Pod 平均排队不超过 20 个请求
```

**速率限制**：HPA 不会无限快地扩容。单次扩容最多翻倍（当前副本数 > 2 时），或一次最多扩到 4 个（副本数 ≤ 2 时）。扩容冷却：3 分钟内不重复扩容；缩容冷却：5 分钟内不重复缩容。这防止了指标抖动时的频繁 thrashing。

**在 AI 推理中的实践**：

CPU 利用率对推理服务不是好的扩缩容指标——LLM 推理主要瓶颈是 GPU，而不是 CPU。更好的指标：
- `inference_queue_length`（推理请求排队长度）：队列积压说明算力不足
- `time_to_first_token_p99`（TTFT P99 延迟）：延迟升高说明需要扩容
- GPU 利用率（通过 DCGM Exporter 暴露）

这些自定义指标需要部署 Prometheus + Prometheus Adapter，然后在 HPA 的 `metrics` 中用 `type: Pods` 或 `type: Object` 引用。

**HPA 的前提**：所有被扩缩容的 Pod 必须设置 CPU requests，否则 HPA 无法计算 CPU 利用率百分比（分母是 requests，不是节点总量）。

### 6. VPA：垂直 Pod 自动扩缩容

VPA（VerticalPodAutoscaler）的思路与 HPA 相反——不增加副本数，而是调整单个 Pod 的 requests 大小。它通过分析历史资源使用模式，推荐或自动设置更合适的 requests 值。

与 HPA 的核心区别：
- HPA：横向扩展，增加副本 → 无需重启 → 适合无状态服务
- VPA：纵向扩展，调整单 Pod 配额 → **需要重启 Pod**（当前实现限制）→ 适合不易水平扩展的单体应用

在写作本书时（2018 年），VPA 仍是实验性功能，需要重启的限制使其在生产环境中使用受限。现代 Kubernetes 中 VPA 已稳定，但"重启才能生效"的限制仍然存在。

### 7. Cluster Autoscaler：节点级弹性

当 Pod 因资源不足而 Pending 时，HPA 无法解决根本问题——节点本身已满。Cluster Autoscaler 负责这一层：

**扩容触发条件**：存在因资源不足无法调度的 Pending Pod。Cluster Autoscaler 检查现有节点组（node group），找到能容纳该 Pod 的节点类型，向云厂商 API 申请新节点。

**缩容触发条件**：节点上所有 Pod 的 requests 总量低于节点 allocatable 的 50%，且这些 Pod 可以迁移到其他节点（不能有 local storage、不能是单例系统 Pod）。缩容时先 cordon（标记不可调度），再 drain（驱逐所有 Pod），然后归还给云厂商。

**PodDisruptionBudget（PDB）与 Cluster Autoscaler 的配合**：

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
spec:
  minAvailable: 2       # 缩容时确保至少 2 个推理副本存活
  selector:
    matchLabels:
      app: vllm-inference
```

PDB 是 Cluster Autoscaler 缩容的安全阀。如果驱逐某节点上的 Pod 会违反 PDB，Cluster Autoscaler 不会缩容该节点，保证服务可用性。

**在 AI 推理中**：GPU 节点昂贵（A100/H100 实例每小时数十美元）。理想状态是：
- 低峰期：只保留 1-2 个 GPU 节点，其余归还
- 高峰期：Cluster Autoscaler 自动扩展 GPU 节点
- 关键约束：GPU 节点启动时间（包括 NVIDIA 驱动初始化）通常 3-5 分钟，需要提前 warm up 或设置合理的 minNodes

---

## Ch16：高级调度——精确控制 Pod 的落点

### 8. Taints 与 Tolerations：排斥与容忍

Taint 是打在**节点**上的标记，表示"我不欢迎没有相应 Toleration 的 Pod"。Toleration 是打在 **Pod** 上的标记，表示"我能容忍某些节点的 Taint"。

这种设计是**节点主动选择**，而非 Pod 主动选择——节点管理员决定哪些节点是专用的，Pod 需要声明自己有资格使用这些节点。

**三种 Taint Effect**：

| Effect | 含义 |
|--------|------|
| `NoSchedule` | 新 Pod 不会被调度到该节点（已运行的 Pod 不受影响） |
| `PreferNoSchedule` | 尽量避免调度到该节点，但实在没地方时仍可以 |
| `NoExecute` | 新 Pod 不调度；且已在节点上运行但不容忍该 taint 的 Pod 被驱逐 |

**GPU 节点的标准做法**：

```bash
# 给 GPU 节点打上 taint，防止普通 Pod 占用 GPU 资源
kubectl taint node gpu-node-a100 nvidia.com/gpu=true:NoSchedule
```

```yaml
# LLM 推理 Pod 必须声明 toleration 才能调度到 GPU 节点
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"     # 只要有这个 key，不管 value
    effect: "NoSchedule"
  containers:
  - name: vllm-server
    resources:
      requests:
        nvidia.com/gpu: "1"
```

**NoExecute + tolerationSeconds**：当节点变为 NotReady 或 Unreachable 时，K8s 自动给节点加上 `NoExecute` taint。Pod 上默认有 300 秒的 `tolerationSeconds`——即节点故障 5 分钟后，Pod 才被驱逐并重新调度。对于延迟敏感的推理服务，可以将这个时间缩短：

```yaml
tolerations:
- key: "node.kubernetes.io/not-ready"
  operator: "Exists"
  effect: "NoExecute"
  tolerationSeconds: 30    # 30 秒后驱逐，更快故障转移
```

### 9. Node Affinity：主动选择节点

与 Taint/Toleration（节点拒绝）不同，Node Affinity 是 **Pod 主动选择**——Pod 声明自己想去哪类节点。

Node Affinity 比旧的 `nodeSelector` 更强大：

**硬约束（Required）**：

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu-type
            operator: In
            values: ["A100", "H100"]   # 必须调度到 A100 或 H100 节点
          - key: availability-zone
            operator: In
            values: ["zone1"]
```

**软约束（Preferred）**：

```yaml
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 80
        preference:
          matchExpressions:
          - key: availability-zone
            operator: In
            values: ["zone1"]    # 优先 zone1，但不是必须
      - weight: 20
        preference:
          matchExpressions:
          - key: gpu-generation
            operator: In
            values: ["hopper"]   # 次要偏好：Hopper 架构 GPU
```

`weight` 字段（1-100）决定偏好的相对重要性。Scheduler 将各偏好规则的权重加到节点的最终得分上。

**`requiredDuringSchedulingIgnoredDuringExecution` 这个长名字的含义**：
- `requiredDuringScheduling`：调度时必须满足
- `IgnoredDuringExecution`：Pod 运行后即便节点标签变化，也不驱逐（未来会支持 `RequiredDuringExecution`）

**对比 nodeSelector**：nodeSelector 只能做精确匹配（`key: value`），Node Affinity 支持 `In`、`NotIn`、`Exists`、`DoesNotExist`、`Gt`、`Lt` 等多种操作符，表达能力强得多。

### 10. Pod Affinity / Anti-Affinity：Pod 间协同调度

有时候重要的不是 Pod 去哪个节点，而是 **Pod 相对于其他 Pod 的位置关系**。

**Pod Affinity（亲和性）**：把相关 Pod 调度在一起，降低通信延迟。

```yaml
# 推理 Pod 希望和缓存 sidecar 调度到同一节点
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: kubernetes.io/hostname  # 同一节点（hostname 相同）
        labelSelector:
          matchLabels:
            app: kv-cache-proxy
```

**Pod Anti-Affinity（反亲和性）**：把同类 Pod 分散到不同节点/Zone，提高可用性。

```yaml
# 推理 Pod 的多个副本必须分散在不同节点
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: kubernetes.io/hostname
        labelSelector:
          matchLabels:
            app: vllm-inference
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          topologyKey: failure-domain.beta.kubernetes.io/zone  # 软约束：跨 AZ
          labelSelector:
            matchLabels:
              app: vllm-inference
```

**topologyKey 的灵活性**：topologyKey 可以是任何节点标签。设为 `kubernetes.io/hostname` 表示同节点粒度；设为 `topology.kubernetes.io/zone` 表示同 AZ 粒度；设为 `rack`（自定义标签）表示同机柜粒度。

**在 AI 推理中的典型用例**：
- 推理 + Prometheus exporter sidecar：podAffinity，同节点，减少网络 RTT
- 多个推理副本：podAntiAffinity + `hostname`，强制分散，单节点故障不影响整体服务
- 训练 Pod（多机多卡）：RDMA 通信需要同机架，podAffinity + `rack`

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/scheduling-pipeline.html" height="520px" title="Scheduler 调度流水线" />

---

## 综合：AI 推理集群的资源调度设计

将本文所有概念综合起来，一个生产级 AI 推理集群的资源调度设计应该是这样的：

```yaml
# 完整的 LLM 推理 Deployment：整合所有资源调度最佳实践
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: ai-team-a
spec:
  replicas: 3
  template:
    spec:
      # ── 1. Taint Toleration：允许调度到 GPU 节点 ──────────────────────
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"

      # ── 2. Node Affinity：要求 A100/H100，优先 zone1 ──────────────────
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: gpu-type
                operator: In
                values: ["A100", "H100"]
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 60
            preference:
              matchExpressions:
              - key: availability-zone
                operator: In
                values: ["zone1"]

        # ── 3. Pod Anti-Affinity：副本分散到不同节点 ──────────────────
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - topologyKey: kubernetes.io/hostname
            labelSelector:
              matchLabels:
                app: vllm-inference

      containers:
      - name: vllm-server
        resources:
          # ── 4. requests == limits = Guaranteed QoS ─────────────────
          requests:
            cpu: "4"
            memory: "32Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "32Gi"
            nvidia.com/gpu: "1"
```

---

## 速查表

| 概念 | 作用层级 | 核心作用 | AI 推理场景 |
|------|---------|---------|------------|
| `requests` | 容器 | Scheduler 调度依据 | 声明 GPU 需求 |
| `limits` | 容器 | 运行时上限，超出被 OOM/throttle | 防止内存溢出 |
| QoS Guaranteed | Pod | 最高优先级，不被轻易驱逐 | 在线推理服务 |
| QoS Burstable | Pod | 中等优先级，可超量使用 | 离线批处理 |
| QoS BestEffort | Pod | 最低优先级，资源压力下最先牺牲 | 开发测试 |
| LimitRange | Namespace | 设置 Pod 资源的默认值和边界 | 防止无配额 Pod |
| ResourceQuota | Namespace | 限制命名空间总资源量 | 多团队 GPU 配额 |
| HPA | Deployment | 按指标自动调整副本数 | 按请求队列扩缩容 |
| VPA | Pod | 自动调整 requests（需重启） | 单体服务资源调优 |
| Cluster Autoscaler | Node | 按负载增删节点 | GPU 节点按需伸缩 |
| Taint/Toleration | Node/Pod | 节点排斥特定 Pod | GPU 节点专用化 |
| Node Affinity | Pod | Pod 主动选择节点类型 | 指定 GPU 型号 |
| Pod Affinity | Pod | Pod 间同位置调度 | 推理+缓存同节点 |
| Pod Anti-Affinity | Pod | Pod 间分散调度 | 副本跨节点高可用 |
| PodDisruptionBudget | Pod 集合 | 限制同时不可用的 Pod 数量 | 缩容时保证服务连续性 |
