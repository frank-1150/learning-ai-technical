---
date: "2026-04-10"
title: "配置管理、密钥安全与应用滚动部署"
description: "ConfigMap 与 Secret 解耦配置，Downward API 注入 Pod 元数据，Deployment 实现零停机更新"
tags: [kubernetes, configmap, secret, deployment, rolling-update, downward-api]
---

> 本文对应原书 **第 7-8-9 章**，覆盖：ConfigMaps and Secrets、Accessing pod metadata from applications、Deployments: updating applications declaratively

# 配置管理、密钥安全与应用滚动部署

到这一章，我们已经知道如何把应用打包成容器、让 Pod 跑起来、通过 Service 对外提供服务。但一个实际的应用还差两件事没解决：**配置从哪里来**，以及**怎么在不停机的情况下发布新版本**。这两个问题，正是第 7-8-9 章的核心。

---

## 一、配置与应用分离的原则

### 为什么不能把配置打进镜像

12-Factor App 第三条明确指出：**将配置存储在环境中**（Store config in the environment）。这里的"配置"是指在不同部署环境（开发、测试、生产）之间会变化的任何东西：数据库地址、API 密钥、并发限制、feature flag……

如果把配置写死在镜像里，会有三个麻烦：

1. **环境差异需要多个镜像**：dev 用一个镜像，prod 用另一个，版本管理立刻混乱。
2. **修改配置 = 重新构建镜像**：一次数据库地址变更就要走完整的 CI 流程，代价极高。
3. **镜像包含敏感信息**：拥有镜像访问权的人都能看到 API Key、数据库密码。

### 三种传递配置的方式

Kubernetes 提供了三种层次的配置传递手段，从简单到复杂依次是：

| 方式 | 适用场景 | 缺点 |
|------|---------|------|
| 命令行参数（args） | 少量简单参数 | 不能运行时修改 |
| 环境变量（env） | 中等复杂度配置 | 不能运行时修改，有 hardcode 风险 |
| ConfigMap / Secret Volume | 完整配置文件、敏感数据 | 略有延迟（最终一致） |

直接在 Pod spec 里写 `env.value: "30"` 虽然可行，但一旦你需要在 dev/prod 用不同值，就必须维护两份 Pod 定义。更好的方式是用 ConfigMap 把配置从 Pod 定义里抽离出来。

---

## 二、ConfigMap：外部化配置的标准做法

### 什么是 ConfigMap

ConfigMap 是 Kubernetes 原生的键值存储对象，值可以是短字符串，也可以是完整的配置文件内容：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fortune-config
data:
  sleep-interval: "25"
  my-nginx-config.conf: |
    server {
      listen 80;
      gzip on;
      gzip_types text/plain application/xml;
      ...
    }
```

ConfigMap 的关键设计是**解耦**：同名的 ConfigMap 在 dev 命名空间和 prod 命名空间可以有完全不同的内容，而 Pod 定义一字不改——它只引用名字，不关心具体值。

![图 7.3 同名 ConfigMap 在不同命名空间中持有不同配置值](/book-notes/kubernetes-in-action/images/fig-7-3-configmap-different-envs.png)

### 创建 ConfigMap 的三种来源

```bash
# 1. 字面量
kubectl create configmap my-config \
  --from-literal=sleep-interval=25 \
  --from-literal=env=production

# 2. 单个文件（文件名作为 key）
kubectl create configmap my-config \
  --from-file=nginx.conf

# 3. 整个目录（每个文件名作为一个 key）
kubectl create configmap my-config \
  --from-file=./configmap-files/
```

### 三种使用方式

**方式一：作为单个环境变量**

```yaml
env:
- name: INTERVAL
  valueFrom:
    configMapKeyRef:
      name: fortune-config
      key: sleep-interval
```

**方式二：一次性暴露所有 key 为环境变量**（使用 `envFrom`）

```yaml
envFrom:
- prefix: CONFIG_
  configMapRef:
    name: my-config-map
```

注意：包含短横线的 key（如 `foo-bar`）不是合法的环境变量名，Kubernetes 会跳过这些 key 并记录一个事件。

**方式三：挂载为 Volume 中的文件**（推荐用于大型配置文件）

```yaml
volumes:
- name: config
  configMap:
    name: fortune-config
    items:
    - key: my-nginx-config.conf
      path: gzip.conf        # 在 volume 内的文件名
volumeMounts:
- name: config
  mountPath: /etc/nginx/conf.d
  readOnly: true
```

![图 7.9 ConfigMap 条目以文件形式挂载到 Pod 的 Volume 中](/book-notes/kubernetes-in-action/images/fig-7-9-configmap-volume-mount.png)

### 热更新：ConfigMap Volume 的独特优势

环境变量和命令行参数在 Pod 创建后**无法修改**。但以 Volume 挂载的 ConfigMap 是个例外——当你 `kubectl edit configmap` 修改内容后，Kubernetes 会在后台（最长约一分钟内）更新所有挂载了该 ConfigMap 的 Pod 中的文件。

Kubernetes 用符号链接实现原子更新：实际文件写入一个时间戳目录，然后把 `..data` 软链接一次性切换到新目录，确保所有文件同时更新，进程不会读到一半新一半旧的状态。

**一个重要警告**：如果你用 `subPath` 挂载了单个文件（而非整个 Volume），该文件**不会**被热更新。

**在 AI 推理服务中的应用**：将 vLLM 或 TGI 的服务参数（`--max-model-len`、`--tensor-parallel-size`、`--max-num-seqs`）放入 ConfigMap，配合支持 SIGHUP 重载的应用层逻辑，可以实现无需重建镜像的参数调整。

---

## 三、Secret：敏感数据的正确处理方式

### 为什么 ConfigMap 不够

ConfigMap 的内容以明文存储在 etcd 中，`kubectl get configmap -o yaml` 可以直接看到所有值。把 OpenAI API Key 或数据库密码放进去是明显的安全问题。

Secret 的设计在几个维度上更安全：

- **内存存储**：Secret Volume 挂载时使用 tmpfs（内存文件系统），数据**不落磁盘**
- **按需分发**：只有需要该 Secret 的节点才会收到它（通过 kubelet）
- **etcd 加密**：从 Kubernetes 1.7 起，etcd 中的 Secret 以加密形式存储

不过需要明确：**Base64 编码不是加密**。`kubectl get secret -o yaml` 看到的 Base64 字符串，解码即可得到原文。Secret 的安全性依赖于 RBAC 权限控制和 etcd 加密，而不是编码本身。

### Secret 的类型

| 类型 | 用途 |
|------|------|
| `Opaque` | 通用，任意键值 |
| `kubernetes.io/tls` | TLS 证书和私钥 |
| `kubernetes.io/dockerconfigjson` | 私有镜像仓库认证 |
| `kubernetes.io/service-account-token` | ServiceAccount Token（自动创建） |

### 创建和使用 Secret

```bash
# 创建包含 TLS 证书的 Secret
kubectl create secret generic fortune-https \
  --from-file=https.key \
  --from-file=https.cert
```

在 Pod 中挂载：

```yaml
volumes:
- name: certs
  secret:
    secretName: fortune-https
volumeMounts:
- name: certs
  mountPath: /etc/nginx/certs/
  readOnly: true
```

验证是否使用 tmpfs：
```bash
kubectl exec fortune-https -c web-server -- mount | grep certs
# tmpfs on /etc/nginx/certs type tmpfs (ro,relatime)
```

![图 7.12 ConfigMap 与 Secret 组合使用的完整 Pod 架构](/book-notes/kubernetes-in-action/images/fig-7-12-configmap-secret-combined.png)

### 通过环境变量暴露 Secret 的风险

虽然技术上可以把 Secret 的值设为环境变量，但书中明确建议**避免这样做**：

- 应用崩溃时往往会把所有环境变量 dump 到日志或错误报告
- 子进程会继承父进程的所有环境变量，难以控制传播范围
- 第三方库可能在某些情况下打印所有环境变量

**最佳实践：始终通过 Volume 挂载来暴露 Secret。**

**在 AI 推理服务中的应用**：
- 外部模型 API 密钥（OpenAI、Anthropic）→ Opaque Secret + Volume
- 私有模型仓库（HuggingFace private repo token）→ Opaque Secret
- 私有容器镜像仓库（NGC, GCR）→ `docker-registry` Secret + `imagePullSecrets`

---

## 四、Downward API：让 Pod 感知自身元数据

### 一个实际需求

配置管理解决了"怎么把外部配置传进来"的问题。但有一类信息是在 Pod 调度之后才确定的：Pod 的名字（`kubia-1234-abcd`）、Pod 运行的节点名、Pod 的 IP 地址、当前 namespace……这些信息在写 YAML 时你根本不知道，但应用有时需要用到它们。

Downward API 是 Kubernetes 的解决方案——它不是一个 REST 接口，而是一种**把 Pod 自身的元数据注入到 Pod 里**的机制，通过环境变量或文件两种方式传递。

### 可以暴露的元数据

- Pod 名称、命名空间、IP 地址
- 所在节点名称
- 使用的 ServiceAccount 名称
- 容器的 CPU/内存 requests 和 limits
- Pod 的 labels 和 annotations（**只能通过 Volume，不能用环境变量**）

### 通过环境变量注入

```yaml
env:
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: CONTAINER_CPU_REQUEST_MILLICORES
  valueFrom:
    resourceFieldRef:
      resource: requests.cpu
      divisor: 1m        # 结果单位为毫核
- name: CONTAINER_MEMORY_LIMIT_KIBIBYTES
  valueFrom:
    resourceFieldRef:
      resource: limits.memory
      divisor: 1Ki       # 结果单位为 KiB
```

![图 8.2 Pod 元数据通过 Downward API 注入为环境变量，值来源于 Pod manifest 和 status](/book-notes/kubernetes-in-action/images/fig-8-2-downward-api-env-vars.png)

### 通过 downwardAPI Volume 注入（支持 labels/annotations 热更新）

```yaml
volumes:
- name: downward
  downwardAPI:
    items:
    - path: "labels"
      fieldRef:
        fieldPath: metadata.labels
    - path: "annotations"
      fieldRef:
        fieldPath: metadata.annotations
volumeMounts:
- name: downward
  mountPath: /etc/downward
```

labels 和 annotations 只能通过 Volume 方式暴露，因为它们可能在 Pod 运行期间被修改，而环境变量创建后无法更新。当 labels/annotations 变化时，Kubernetes 会自动更新 Volume 中对应的文件内容。

**在 AI 推理服务中的应用**：用 `POD_NAME` 作为分布式追踪的实例 ID（trace instance ID），配合 OpenTelemetry，可以精确定位哪个 Pod 处理了哪个请求，对于排查推理服务的长尾延迟非常有帮助。

---

## 五、与 Kubernetes API Server 通信

### 为什么需要直接访问 API Server

Downward API 只能暴露 Pod 自身的有限元数据。如果应用需要动态发现集群中的其他 Pod、监控 Service 的 Endpoint 变化、或者动态创建/删除资源，就必须直接调用 Kubernetes API Server。

### 从 Pod 内部访问 API Server

三件事需要处理：

1. **找到 API Server 地址**：Kubernetes 自动在每个 Pod 中注入 `KUBERNETES_SERVICE_HOST` 和 `KUBERNETES_SERVICE_PORT` 环境变量，或者直接用 `https://kubernetes` DNS 名称。

2. **验证服务器身份**：使用自动挂载在 `/var/run/secrets/kubernetes.io/serviceaccount/ca.crt` 的 CA 证书验证 API Server 的 TLS 证书。

3. **认证自己**：使用 `/var/run/secrets/kubernetes.io/serviceaccount/token` 中的 Bearer Token。

```bash
# 从 Pod 内部访问 API Server
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
NS=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
curl -H "Authorization: Bearer $TOKEN" \
     --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
     https://kubernetes/api/v1/namespaces/$NS/pods
```

### Ambassador Container 模式

直接处理 HTTPS 认证对应用开发者来说负担较重，尤其是用非主流语言开发的服务。Ambassador 模式提供了一个优雅的解法：在 Pod 中运行一个 `kubectl proxy` 容器作为 sidecar，主容器只需发 HTTP 请求到 `localhost:8001`，由 Ambassador 容器透明地处理 HTTPS 连接、CA 验证和 Token 注入。

```
主容器  --HTTP--> Ambassador(kubectl proxy) --HTTPS+Token--> API Server
```

因为同一 Pod 中的容器共享网络命名空间，主容器通过 loopback 就能访问 Ambassador。这种模式特别适合需要操作 Kubernetes 资源的运维类应用（如 operator sidecar）。

---

## 六、Deployment：声明式滚动更新

### 命令式 rolling-update 的问题

在 Deployment 出现之前，更新应用要用 `kubectl rolling-update` 命令：

```bash
kubectl rolling-update kubia-v1 kubia-v2 --image=luksa/kubia:v2
```

这个命令实际上是**由 kubectl 客户端**驱动的——kubectl 不断向 API Server 发送 PUT 请求，一步一步地缩减旧 RC、扩展新 RC。问题显而易见：

- **网络中断 = 更新卡住**：如果 kubectl 进程在更新途中断开，集群会停在一个中间状态
- **命令式思维**：你在告诉 Kubernetes "怎么做"，而不是"要什么结果"
- **不透明**：Kubernetes 被迫修改你创建的对象（给 Pod 加 label、改 RC selector），违背了 K8s 的设计哲学

### Deployment：声明式 + 服务端执行

Deployment 把决策权还给了集群：你只需要更新 Pod template 里的镜像标签，Kubernetes 控制面（controller manager）自动完成后续的一切。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kubia
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: nodejs
        image: luksa/kubia:v1
```

触发更新只需一行：
```bash
kubectl set image deployment kubia nodejs=luksa/kubia:v2
```

或者修改 YAML 后 apply：
```bash
kubectl apply -f kubia-deployment-v2.yaml
```

### Deployment 如何管理 ReplicaSet

Deployment 并不直接管理 Pod，而是通过 ReplicaSet：

```
Deployment --> ReplicaSet v1 (desired: 0, current: 0)
           --> ReplicaSet v2 (desired: 3, current: 3)
```

每个 ReplicaSet 的名字包含 Pod template 的哈希值，这让 Deployment 可以为每个版本复用已有的 ReplicaSet（如果 Pod template 回滚到之前的版本）。

**关键设计**：旧的 ReplicaSet 在更新后并不会被删除，它保留着 0 个副本，这正是实现回滚的基础——它就是版本历史本身。

### 两种更新策略

**RollingUpdate（默认）**：逐步替换，全程维持服务可用

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 最多允许比 desired 多 1 个 Pod
      maxUnavailable: 0  # 不允许任何 Pod 不可用
```

**Recreate**：先删除所有旧 Pod，再创建新 Pod。适用于不支持两个版本同时运行的应用（如有数据库 schema 迁移的场景）。

### maxSurge 与 maxUnavailable 的含义

以 `replicas: 3, maxSurge: 1, maxUnavailable: 0` 为例，整个更新流程如下：

1. 创建 1 个 v2 Pod（此时共 4 个 Pod：3v1 + 1v2）
2. 等待 v2 Pod Ready
3. 删除 1 个 v1 Pod（恢复到 3 个：2v1 + 1v2）
4. 创建 1 个 v2 Pod → 等待 Ready → 删除 1 个 v1
5. 循环直到全部替换完成

`maxUnavailable: 0` 确保任何时刻都有 3 个可用 Pod，代价是需要额外的资源（第 4 个 Pod）。

![图 9.10 Deployment 滚动更新前后的 ReplicaSet 状态](/book-notes/kubernetes-in-action/images/fig-9-10-deployment-rolling-update.png)

### 回滚：kubectl rollout undo

```bash
# 回滚到上一个版本
kubectl rollout undo deployment kubia

# 查看版本历史（需创建时加 --record）
kubectl rollout history deployment kubia

# 回滚到指定版本
kubectl rollout undo deployment kubia --to-revision=1
```

每个历史版本对应一个 ReplicaSet，版本历史的长度由 `revisionHistoryLimit` 控制（默认 2，即保留当前和上一版本）。

### 金丝雀发布：暂停与恢复

Deployment 支持在发布过程中暂停，这自然地实现了金丝雀部署模式：

```bash
# 触发更新后立即暂停
kubectl set image deployment kubia nodejs=luksa/kubia:v4
kubectl rollout pause deployment kubia

# 此时只有 1 个 v4 Pod 运行，其余仍是旧版本
# 观察一段时间，确认 v4 行为正常...

# 继续完成更新
kubectl rollout resume deployment kubia

# 或者回滚
kubectl rollout undo deployment kubia
```

### 用 minReadySeconds + readinessProbe 自动阻止坏发布

这是 Deployment 最重要但容易被忽视的安全机制：

```yaml
spec:
  minReadySeconds: 10        # Pod 就绪后必须保持就绪 10 秒，才算真正可用
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: nodejs
        readinessProbe:
          httpGet:
            path: /
            port: 8080
          periodSeconds: 1   # 每秒检测一次
```

如果新版本的 readiness probe 在 `minReadySeconds` 内开始失败，新 Pod 永远不会被认为"可用"，`maxUnavailable: 0` 又不允许缩减旧 Pod，更新进程就此停滞——这是一个保护机制，阻止了错误版本的全量发布。

超过 `progressDeadlineSeconds`（默认 10 分钟）后，Deployment 会被标记为 `ProgressDeadlineExceeded`，此时需要手动 `kubectl rollout undo` 终止。

**在 AI 推理服务中的应用**：
- 模型版本 v1 → v2 的滚动更新，通过 `maxUnavailable: 0` 确保推理服务全程可用
- `minReadySeconds` 设置为 120 秒（模型加载通常需要几十秒到几分钟），确保模型完全加载后才开始接收流量
- readiness probe 检测 `/health` 或首个推理请求，验证模型确实可用
- 金丝雀发布：先更新 1 个副本，观察 P99 延迟和错误率，再全量推进

---

## 七、交互式可视化：Deployment 滚动更新动画

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/deployment-rollout.html" height="500px" title="Deployment 滚动更新动画" />

---

## 总结对比表

| 概念 | 本质 | 核心用途 | 注意事项 |
|------|------|---------|---------|
| ConfigMap | K8s 键值存储对象 | 外部化非敏感配置 | Volume 挂载支持热更新；subPath 挂载不支持 |
| Secret | 加密存储，内存挂载 | 敏感数据（密钥、证书） | Base64 != 加密；优先 Volume 而非 env var |
| Downward API | Pod 自身元数据注入机制 | 让应用感知自身环境 | labels/annotations 只能用 Volume 方式 |
| API Server 访问 | REST API + Bearer Token | 动态发现/操作集群资源 | Ambassador 模式简化认证 |
| Deployment | ReplicaSet 上层抽象 | 声明式滚动更新 | 服务端执行，不依赖 kubectl 持续在线 |
| RollingUpdate | 逐步替换策略 | 零停机更新 | maxSurge/maxUnavailable 控制速率 |
| rollout undo | 基于旧 ReplicaSet 回滚 | 快速回退坏版本 | --record 保证历史可读 |
| 金丝雀发布 | pause/resume 机制 | 渐进式流量切换 | 目前需手动控制比例 |
| minReadySeconds | 延迟"可用"判定 | 防止不稳定版本扩散 | 须配合 readinessProbe 才有意义 |
