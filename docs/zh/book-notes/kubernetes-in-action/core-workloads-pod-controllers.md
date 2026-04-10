---
date: "2026-04-10"
title: "核心工作负载：Pod 与副本控制"
description: "深入理解 Pod 的设计哲学，以及 ReplicaSet、DaemonSet、Job 如何管理工作负载生命周期"
tags: [kubernetes, pod, replicaset, daemonset, job, cronjob]
---

# 核心工作负载：Pod 与副本控制

> 本文对应原书 **第 3-4 章**，覆盖：Pods: running containers in Kubernetes、Replication and other controllers: deploying managed pods

---

## 一、Pod：K8s 的最小调度单元

### 为什么不直接调度容器？

这是理解 K8s 架构的第一个关键问题。答案在于**进程与容器的关系**。

Docker 的设计哲学是：每个容器运行一个进程。这很好，但现实中的应用往往由多个紧密协作的进程组成——例如一个 Web 服务 + 一个日志轮转守护进程，或者一个主推理进程 + 一个指标采集 sidecar。这些进程需要共享文件系统或通过 localhost 通信，必须运行在同一台机器上。

把所有进程塞进一个容器？这违背了容器的隔离设计，且让日志、重启策略、依赖管理都变得混乱。

**Pod 是这个问题的解法**：它是一个或多个容器的集合，这些容器：
- 永远运行在同一个 Node 上（不会跨节点分散）
- 共享同一个 **Network namespace**（相同 IP 和端口空间，可通过 localhost 互通）
- 共享同一个 **UTS namespace**（相同 hostname）
- 共享同一个 **IPC namespace**（可通过 IPC 互通）
- 文件系统默认隔离，但可通过 Volume 挂载共享

书中 Figure 3.1 展示了这个约束的直观含义：Pod 内的容器必须同节点，容器跨 Pod 则天然分布在不同节点上。

### Pod 的扁平网络模型

K8s 集群中所有 Pod 共享一个"平坦"的网络地址空间（Figure 3.2）：
- 每个 Pod 有独立的 IP 地址，在集群内全局可达
- Pod 之间通信**不需要 NAT**，直接走 IP
- 这一网络通常通过软件定义网络（SDN，如 Flannel、Calico）叠加在物理网络之上

这个设计让 Pod 的网络行为非常接近物理机或虚拟机，跨节点通信就像在同一个局域网。

### Pod 的 YAML 结构解析

一个 K8s 资源的 YAML 通常包含三大部分：

```yaml
apiVersion: v1          # API 版本
kind: Pod               # 资源类型
metadata:               # 元数据（名称、命名空间、标签、注解）
  name: inference-pod
  labels:
    app: llm-inference
    env: production
spec:                   # 规格（期望状态）
  containers:
  - name: inference
    image: vllm/vllm-openai:latest
    ports:
    - containerPort: 8000
  - name: metrics-exporter
    image: prom/pushgateway:latest
    ports:
    - containerPort: 9091
status:                 # 当前状态（运行时填充，创建时不需要写）
  phase: Running
```

`metadata` 和 `spec` 是创建资源时需要填写的，`status` 由 Kubernetes 自动维护，反映资源的当前真实状态。

### AI 推理服务中的 Sidecar 模式

书中 Figure 3.3 展示了 Pod 中"一主多辅"的经典模式。在 AI 基础设施场景，这个模式极其常见：

**典型场景：vLLM 推理服务 + DCGM 指标采集**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-inference-pod
spec:
  containers:
  - name: vllm-server           # 主容器：推理服务
    image: vllm/vllm-openai:latest
    args: ["--model", "meta-llama/Llama-3-8B-Instruct"]
    ports:
    - containerPort: 8000
    resources:
      limits:
        nvidia.com/gpu: "1"
    volumeMounts:
    - name: shared-metrics
      mountPath: /tmp/metrics

  - name: metrics-exporter      # Sidecar：指标采集
    image: prom/pushgateway:latest
    ports:
    - containerPort: 9091
    volumeMounts:
    - name: shared-metrics
      mountPath: /tmp/metrics

  volumes:
  - name: shared-metrics
    emptyDir: {}
```

主推理容器将运行时指标写入共享 Volume，sidecar 容器读取并推送到 Prometheus。两个容器通过 `localhost` 互通，共享相同的生命周期——这正是 Pod 设计意图的体现。

### Labels 与 Selectors：K8s 最强大的组织机制

Label 是 Pod（及所有 K8s 资源）上的任意 key-value 对，例如 `app=llm-inference`、`env=production`、`version=v2`。

书中 Figure 3.7 展示了在微服务架构中用两个维度的 label（`app` 和 `rel`）组织数十个 Pod 的效果——从混乱的 Figure 3.6 到清晰的多维矩阵。

**Label Selector 的威力**：不只是 `kubectl get pods -l app=kubia`，更重要的是 K8s 系统内部用 selector 来关联资源：
- ReplicaSet 通过 selector 知道管理哪些 Pod
- Service 通过 selector 决定将流量路由到哪些 Pod
- DaemonSet 通过 nodeSelector 决定部署到哪些节点

**四种 matchExpressions 操作符**（ReplicaSet 引入）：
- `In`：label 值在指定列表中
- `NotIn`：label 值不在指定列表中
- `Exists`：label key 存在（不管值）
- `DoesNotExist`：label key 不存在

在 AI 集群中，用 `accelerator=a100` 或 `accelerator=h100` 给节点打 label，再用 nodeSelector 让推理 Pod 精准落在 GPU 节点，是非常典型的用法：

```yaml
spec:
  nodeSelector:
    gpu: "true"
    accelerator: h100
```

### Namespaces：多租户资源隔离

Namespace 将集群内的资源分组到互不重叠的命名空间中。同一 namespace 内资源名唯一，不同 namespace 可以有同名资源。

典型用途：
- `default`：日常开发测试
- `production` / `staging`：环境隔离
- `kube-system`：K8s 系统组件（DNS、kube-proxy 等）
- `monitoring`：Prometheus、Grafana 等监控组件

**重要提示**：Namespace 提供的是资源管理隔离，**不是网络隔离**。不同 namespace 的 Pod 默认仍可互相通信（除非部署了 NetworkPolicy）。

---

## 二、Liveness Probe：自愈能力的基础

### 为什么不能只依赖进程存活？

Kubernetes 的 Kubelet 会在容器进程崩溃时自动重启容器，这处理了最简单的故障情形。但现实中的应用故障往往不是进程崩溃：

- Java 应用出现内存泄漏，JVM 进程还在但 OOM 不断
- 应用进入死锁，进程健在但无法处理任何请求
- 依赖服务不可用，应用陷入无限等待循环

这些情况进程都没有退出，Kubelet 无法感知。**Liveness Probe** 就是从外部检查应用健康状态的机制。

### 三种探针类型

| 探针类型 | 机制 | 适用场景 |
|---------|------|---------|
| **HTTP GET** | 向指定端口+路径发 GET 请求，2xx/3xx 为健康 | Web 服务、REST API |
| **TCP Socket** | 尝试建立 TCP 连接，成功为健康 | 数据库、非 HTTP 服务 |
| **Exec** | 在容器内执行命令，exit code 为 0 则健康 | 需要自定义逻辑的检查 |

**实际配置示例**（带推荐参数）：

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30   # 重要！给应用充足的启动时间
  periodSeconds: 10         # 每 10 秒检查一次
  timeoutSeconds: 5         # 超时时间
  failureThreshold: 3       # 连续 3 次失败才重启
```

**最重要的实践建议**：`initialDelaySeconds` 不可忽略。推理服务加载大模型权重可能需要数分钟，如果探针在启动后立即开始检测，会触发误重启，导致永远无法正常启动。

对于 AI 推理服务（如 vLLM），`/health` 路径应检查模型是否已加载完成，而不仅仅是服务进程是否存活。

---

## 三、ReplicationController vs ReplicaSet：控制循环的力量

### 控制循环原理：声明式的魔法

这是理解整个 K8s 系统的核心思想。书中 Figure 4.2 展示了 ReplicationController（RC）的协调循环：

```
Start
  ↓
找到所有匹配 label selector 的 Pod
  ↓
比较实际数量 vs 期望数量
  ↙        ↓        ↘
太少      刚好      太多
  ↓                  ↓
创建新 Pod         删除多余 Pod
  ↓                  ↓
  ←←←←←←←←←←←←←←←←
```

RC 不关心"做什么动作"，只关心"当前状态是否符合期望"。这就是**声明式系统**的本质——你声明目标状态，系统自动协调。

### RC 的三个核心要素（Figure 4.3）

1. **Label Selector**：确定 RC 管辖哪些 Pod
2. **Replica Count**：期望运行的 Pod 数量
3. **Pod Template**：创建新 Pod 时使用的模板（类似"饼干模具"）

关键理解：**Pod 并不属于 RC**。RC 通过 label selector 关联 Pod，但 Pod 和 RC 是松耦合的。你可以修改一个 Pod 的 label，让它脱离 RC 的管辖（RC 会新建一个补位的 Pod），然后独立调试这个"逃逸"的 Pod，不会影响生产流量。

### ReplicaSet：RC 的进化版

书中 4.3 节明确指出：ReplicationController 已过时，应始终使用 **ReplicaSet**。两者的关键差别在于 label selector 能力：

| 特性 | ReplicationController | ReplicaSet |
|------|----------------------|------------|
| 等值 selector | ✓ `app=kubia` | ✓ |
| 集合 selector | ✗ | ✓ `matchExpressions` |
| 基于 key 存在匹配 | ✗ | ✓ `Exists` / `DoesNotExist` |
| 多值匹配 | ✗ | ✓ `In` / `NotIn` |

ReplicaSet 的 YAML：

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: llm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
    matchExpressions:
    - key: env
      operator: In
      values: [production, staging]
  template:
    metadata:
      labels:
        app: llm-inference
        env: production
    spec:
      containers:
      - name: inference
        image: vllm/vllm-openai:latest
```

实际使用中，你很少直接创建 ReplicaSet，而是通过 **Deployment**（第 9 章）来管理——Deployment 在 ReplicaSet 之上增加了滚动更新能力。

---

## 四、DaemonSet：每节点一个 Pod

### 设计意图

ReplicaSet 和 ReplicationController 把 Pod 调度到集群中任意合适的节点，整体副本数固定。但有一类工作负载需要不同的语义：**在每个节点上运行且只运行一个实例**。

书中 Figure 4.8 清晰对比了 ReplicaSet（集群整体 5 个 Pod，分布不均）和 DaemonSet（每节点精确 1 个 Pod）的差异。

典型用途：
- **日志收集**：Fluentd / Filebeat，收集节点上所有容器的日志
- **监控 Agent**：Node Exporter，采集节点 CPU/内存/磁盘指标
- **网络插件**：kube-proxy，在每个节点上维护 iptables/ipvs 规则
- **存储守护进程**：Ceph OSD、GlusterFS

### AI 集群中的 DCGM Exporter

在 GPU 集群中，**DCGM Exporter**（NVIDIA DCGM 的 Prometheus 指标采集器）是 DaemonSet 的典型用例。

每个 GPU 节点需要运行一个 DCGM Exporter 实例来采集该节点的 GPU 指标（温度、利用率、显存占用、ECC 错误等），这天然就是 DaemonSet 的适用场景：

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      nodeSelector:
        accelerator: nvidia-gpu    # 只在 GPU 节点上运行
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
        ports:
        - containerPort: 9400
          name: metrics
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]
```

DaemonSet 的一个关键特性：它**绕过调度器**（Scheduler），直接将 Pod 分配到节点，包括那些被标记为不可调度（unschedulable）的节点。这确保了系统级守护进程即使在节点被设置为不接受新 Pod 的情况下也能运行。

---

## 五、Job 和 CronJob：批处理与定时任务

### Job：有终点的任务

ReplicaSet 和 DaemonSet 管理的是**持续运行**的服务，容器退出即重启。Job 管理的是**批处理任务**：容器成功完成（exit code 0）即结束，不重启。

书中 Figure 4.10 展示了 Job 与普通 Pod 的行为差异：当节点故障，Job 管理的 Pod 会被重新调度到其他节点继续完成任务，而直接创建的 Pod 则永久丢失。

**Job 的关键配置参数**：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-finetuning
spec:
  completions: 1          # 需要成功完成的 Pod 数量
  parallelism: 1          # 同时运行的 Pod 数量
  activeDeadlineSeconds: 86400  # 任务超时时间（24小时）
  backoffLimit: 3         # 最大重试次数（默认 6）
  template:
    spec:
      restartPolicy: OnFailure   # Job 必须设置，不能用默认的 Always
      containers:
      - name: trainer
        image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
        command: ["python", "train.py", "--epochs", "100"]
        resources:
          limits:
            nvidia.com/gpu: "8"
```

`restartPolicy` 必须是 `OnFailure` 或 `Never`，不能是 `Always`（默认值），否则 Job 完成后会不断重启。

### 并行 Job：加速大规模批处理

```yaml
spec:
  completions: 100    # 总共需要 100 个 Pod 完成
  parallelism: 10     # 同时最多运行 10 个 Pod
```

这在 AI 场景中非常实用，例如对 10 万条数据进行批量推理时，可以将数据分片，用 100 个 Job Pod 并行处理，每次最多 10 个并行。

### CronJob：定时批处理

CronJob 按 cron 格式的计划定期创建 Job 资源：

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-model-eval
spec:
  schedule: "0 2 * * *"     # 每天凌晨 2 点运行
  startingDeadlineSeconds: 300  # 错过计划时间后的最大容忍延迟
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: evaluator
            image: my-eval:latest
            command: ["python", "run_evals.py"]
```

**AI 场景中的典型 CronJob**：
- 每日凌晨运行模型评估基准测试（benchmark）
- 定期从数据仓库拉取新数据，触发增量微调
- 定期清理过期的模型版本和推理缓存
- 每小时统计推理服务的 token 吞吐量并写入监控数据库

---

## 六、工作负载控制器对比总览

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/pod-controllers-comparison.html" height="520px" title="K8s 工作负载控制器对比" />

| 控制器 | 副本策略 | 完成语义 | 典型 AI 基础设施用例 |
|--------|---------|---------|-------------------|
| **ReplicaSet** | 固定副本数，任意节点 | 持续运行，崩溃即重启 | vLLM 推理服务集群（多副本负载均衡） |
| **DaemonSet** | 每节点 1 个，全量覆盖 | 持续运行，崩溃即重启 | DCGM Exporter（GPU 指标）、Fluentd（日志） |
| **Job** | 一次性，完成即停止 | 成功退出即完成 | 模型微调训练、批量推理、数据预处理 |
| **CronJob** | 定期触发 Job | 每次 Job 完成即结束 | 每日评估基准、定期数据清洗、定期报告生成 |
| **StatefulSet** | 固定副本数，有序+稳定标识 | 持续运行 | 向量数据库（Milvus、Weaviate）、分布式训练参数服务器 |
| **Deployment** | 管理 ReplicaSet，支持滚动更新 | 持续运行 | 推理服务版本升级（蓝绿/金丝雀部署） |

---

## 七、本章关键洞察

从第 3-4 章提炼出的几个认知升级点：

1. **Pod 是调度单元，不是容器**：K8s 永远以 Pod 为粒度调度，这一设计使 sidecar 模式成为一等公民，也是 Service Mesh（Istio）的基础。

2. **控制器不拥有 Pod**：RC/ReplicaSet 通过 label selector 松耦合地管理 Pod。这个设计允许你在不影响生产流量的情况下，把故障 Pod 从控制器"摘出来"调试。

3. **声明式比命令式更强大**：`kubectl scale --replicas=10` 本质上是修改期望副本数字段，触发协调循环。K8s 里你永远在描述"期望状态"，控制器负责达到它。

4. **DaemonSet 绕过调度器**：这是系统级守护进程（GPU 监控、网络插件）能在所有节点可靠运行的保证，即使节点被标记为不可调度。

5. **Job 的语义是完成，不是持续**：`restartPolicy: OnFailure` 是 Job 的关键配置，确保任务完成后不会被无限重启。
