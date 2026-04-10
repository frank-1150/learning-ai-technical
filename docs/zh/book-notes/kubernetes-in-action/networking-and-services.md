---
date: "2026-04-10"
title: "网络与服务发现：流量如何到达你的 Pod"
description: "深入理解 Kubernetes Service 的类型与工作原理，以及 Ingress 如何统一管理外部流量入口"
tags: [kubernetes, service, ingress, networking, load-balancer]
---

# 网络与服务发现：流量如何到达你的 Pod

> 本文对应原书 **第 5 章**，覆盖：Services: enabling clients to discover and talk to pods

Pod 是 Kubernetes 的基本调度单元，但它有一个根本性的问题：**IP 不稳定**。每次 Pod 重建，IP 就会变化。如果你的服务 A 直接把服务 B 的 Pod IP 写死在配置里，那么服务 B 一旦重启，A 就找不到它了。

这正是 Kubernetes **Service** 存在的原因。本章围绕一个核心问题展开：**流量如何可靠地到达你的 Pod？**

---

## 1. 为什么需要 Service

### Pod IP 是动态的

书中总结了三个根本原因，说明为什么不能直接依赖 Pod IP：

1. **Pod 是临时的**（ephemeral）——Pod 可能因节点压力、滚动升级、健康检查失败等原因随时被删除重建；
2. **IP 在调度时才确定**——Pod 的 IP 在被调度到节点之后、启动之前才会分配，客户端无法提前知道；
3. **水平扩展意味着多个 IP**——当你有 3 个副本时，客户端不应该自己维护这张列表，它们只需要一个稳定的地址。

### Service 提供了什么

**Service** 是一个稳定的虚拟端点：

- **固定的 ClusterIP**：一个虚拟 IP，在 Service 的整个生命周期内不变
- **DNS 名称**：`<service-name>.<namespace>.svc.cluster.local`，Pod 通过 DNS 发现服务
- **负载均衡**：kube-proxy 将流量分发到所有健康的后端 Pod

图 5.1 展示了典型的前后端场景：外部客户端通过 Frontend Service 访问多个前端 Pod，前端 Pod 通过 Backend Service 访问后端 Pod。每个 Service 地址固定，不论 Pod 如何漂移。

![图 5.1 — 前后端服务架构](/book-notes/kubernetes-in-action/images/fig-5-1-services-overview.png)

### Service 如何选择 Pod

Service 通过 **标签选择器**（Label Selector）来决定哪些 Pod 属于它。当你创建一个带 `selector: app=kubia` 的 Service，所有带有该标签的 Pod 都会成为它的后端。底层维护一个叫 **Endpoints** 的资源，记录所有匹配 Pod 的 IP:Port 列表。

```yaml
# 最简单的 Service 定义
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  ports:
  - port: 80           # Service 对外暴露的端口
    targetPort: 8080   # 转发到 Pod 的端口
  selector:
    app: kubia         # 匹配所有 app=kubia 的 Pod
```

> [!tip] AI 推理场景
> 在 vLLM 推理集群中，调用方（API Gateway、其他微服务）不应该知道当前是哪个 vLLM Pod 在处理请求。通过 Service，调用方只需请求 `http://vllm-service/v1/completions`，负载均衡由 kube-proxy 透明处理。这也意味着 vLLM Pod 可以随时滚动升级，调用方无感。

---

## 2. Service 的四种类型

Kubernetes 提供四种 Service 类型，满足不同的访问场景：

| 类型 | 访问范围 | 适用场景 |
|------|---------|---------|
| **ClusterIP** | 仅集群内部 | 微服务间调用、内部 API 访问（默认类型） |
| **NodePort** | 节点 IP + 固定端口 | 开发测试、没有云厂商 LB 的裸机环境 |
| **LoadBalancer** | 云厂商外部 IP | 生产环境对外暴露服务 |
| **ExternalName** | 集群内访问外部服务 | 把外部域名包装成集群内可用的服务名 |

### ClusterIP（默认）

最常用的类型。Service 获得一个只在集群内有意义的虚拟 IP，外部无法直接访问。

```yaml
apiVersion: v1
kind: Service
spec:
  type: ClusterIP    # 可以省略，这是默认值
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

Service IP 实际上是一个"虚拟" IP——它不对应任何网络接口，无法被 ping 通。它只在与端口组合时才有意义，由 kube-proxy 的 iptables/IPVS 规则拦截处理。

### NodePort

NodePort 在每个集群节点上开放一个固定端口（范围 30000-32767），外部客户端可以通过任意节点的 IP 加该端口访问服务。

```yaml
apiVersion: v1
kind: Service
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30123   # 不指定则随机分配
  selector:
    app: kubia
```

创建后，以下地址都能访问到这个 Service：
- `10.11.254.223:80`（通过 ClusterIP）
- `130.211.97.55:30123`（通过 Node 1 的外部 IP）
- `130.211.99.206:30123`（通过 Node 2 的外部 IP）

NodePort 的缺点：如果你把客户端固定指向某个节点，该节点宕机后服务就不可用。这就是为什么通常需要在 NodePort 前再架一个负载均衡器。

![图 5.6 — NodePort 流量路径](/book-notes/kubernetes-in-action/images/fig-5-6-nodeport-flow.png)

### LoadBalancer

LoadBalancer 是 NodePort 的扩展。在支持云基础设施的环境（GKE、EKS、AKS 等）中，Kubernetes 会自动向云厂商申请一个外部负载均衡器，并将其 IP 写入 Service 的 `EXTERNAL-IP` 字段。

```yaml
apiVersion: v1
kind: Service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

创建后：

```
NAME               CLUSTER-IP       EXTERNAL-IP      PORT(S)
kubia-loadbalancer 10.111.241.153   130.211.53.173   80:32143/TCP
```

客户端直接访问 `130.211.53.173:80`，无需关心节点 IP 或端口。

![图 5.7 — LoadBalancer 流量路径](/book-notes/kubernetes-in-action/images/fig-5-7-loadbalancer-flow.png)

> **注意**：LoadBalancer service 底层仍然是 NodePort，只是在前面加了一层云厂商 LB。如果云环境不支持，Service 会退化为 NodePort 行为。

### ExternalName

ExternalName 类型不选择 Pod，而是为外部域名创建一个 DNS CNAME 别名。

```yaml
apiVersion: v1
kind: Service
spec:
  type: ExternalName
  externalName: api.somecompany.com
```

集群内的 Pod 可以通过 `external-service.default.svc.cluster.local` 访问外部的 `api.somecompany.com`。这让你可以随时切换后端实现：只需改 `externalName`，所有调用方无感知。

---

## 3. Service 如何工作

### kube-proxy 与 Endpoints 对象

Service 的核心机制由两个组件协作完成：

**Endpoints 对象**：Service 创建时，Kubernetes 会同步创建一个同名的 Endpoints 资源，记录所有匹配 Pod 的 IP:Port：

```
$ kubectl get endpoints kubia
NAME    ENDPOINTS                                         AGE
kubia   10.108.1.4:8080,10.108.2.5:8080,10.108.2.6:8080  1h
```

当 Pod 新增、删除或状态变化时，Endpoints 会自动更新。

**kube-proxy**：运行在每个节点上，监听 Service 和 Endpoints 的变化，并在节点的 iptables（或 IPVS）中写入转发规则。当一个 Pod 发起连接到 Service ClusterIP 时，iptables 规则会将该连接 DNAT（目标地址转换）到某个后端 Pod 的真实 IP:Port。

这意味着：**Service IP 的路由完全发生在节点内核态**，不经过任何中间代理进程。这是 Kubernetes 网络性能的关键所在。

图 5.3 展示了 `kubectl exec` 在 Pod 内运行 curl 命令时，流量经过 Service 转发到随机 Pod 的完整路径：

![图 5.3 — Service 流量转发流程](/book-notes/kubernetes-in-action/images/fig-5-3-kubectl-exec-curl.png)

### Service 发现：环境变量与 DNS

Pod 如何知道 Service 的 ClusterIP 和端口？Kubernetes 提供了两种机制：

**环境变量**：Pod 启动时，Kubernetes 自动注入当前命名空间内所有 Service 的环境变量：
```
KUBIA_SERVICE_HOST=10.111.249.153
KUBIA_SERVICE_PORT=80
```
限制：Service 必须在 Pod 之前创建，否则环境变量不会被注入。

**DNS（推荐）**：每个 Service 在 kube-dns 中自动获得一个 DNS 条目。在同一命名空间内，Pod 可以通过服务名直接访问：
```bash
curl http://kubia          # 同命名空间简写
curl http://kubia.default  # 带命名空间
curl http://kubia.default.svc.cluster.local  # 完整 FQDN
```

DNS 方式更灵活，不依赖启动顺序，是生产环境的首选。

### 一个有趣的陷阱：为什么 ping Service IP 不通

新手常见的误区：Service 的 ClusterIP 是虚拟 IP，**只有与端口组合时才有意义**。你可以 `curl http://10.111.249.153`，但 `ping 10.111.249.153` 会 100% 丢包——因为 iptables 规则只处理 TCP/UDP 连接，不处理 ICMP 报文。

---

## 4. Readiness Probe：精细控制流量

### 问题：Pod Running ≠ 应用就绪

当 Kubernetes 把一个 Pod 启动后，它可能立即加入 Service 的 Endpoints 列表——即便应用还没完成初始化。这会导致：

- 应用刚启动时收到请求 → 返回 500 或连接拒绝
- 依赖加载耗时的应用（数据库连接池初始化、配置文件读取）在就绪前就收到流量

**Readiness Probe** 解决这个问题：它让 Pod 自己告诉 Kubernetes"我准备好了"。

### 三种 Readiness Probe 类型

与 Liveness Probe 一样，Readiness Probe 也有三种检测方式：

| 类型 | 机制 | 适用场景 |
|------|------|---------|
| **Exec** | 执行命令，检查退出码 | 需要复杂条件判断时 |
| **HTTP GET** | 发送 HTTP 请求，检查状态码 | Web 服务的 `/health` 端点 |
| **TCP Socket** | 尝试建立 TCP 连接 | 非 HTTP 服务 |

### Readiness vs. Liveness：关键区别

这是很多人混淆的地方：

- **Liveness Probe 失败** → 容器被 **杀死并重启**
- **Readiness Probe 失败** → 容器从 Service 的 Endpoints 中 **移除**，不接收流量，但**不会重启**

当 Pod 的 Readiness Probe 失败时，Pod 从 Endpoints 列表中剔除：

![图 5.11 — Readiness Probe 失败的 Pod 从 Endpoints 中移除](/book-notes/kubernetes-in-action/images/fig-5-11-readiness-probe.png)

一旦 Probe 恢复成功，Pod 被重新加入 Endpoints，流量自动流入。

### 配置示例

```yaml
spec:
  containers:
  - name: kubia
    image: luksa/kubia
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 5   # 启动后等待 5 秒再开始检查
      periodSeconds: 10        # 每 10 秒检查一次
      failureThreshold: 3      # 连续 3 次失败才移出 Endpoints
```

> [!tip] AI 推理场景
> 这在 AI 推理服务中极其重要。vLLM 加载模型权重（Llama-3-70B 约 140GB）可能需要数分钟。如果没有 Readiness Probe，Pod 刚启动就收到推理请求，会立即报错。正确做法是在 `/health` 端点中检查模型是否已加载完成，只有加载完毕才返回 200。这样 Kubernetes 会等到模型就绪才把流量打过来。

> [!warning] 最佳实践
> 书中明确建议：**永远要配置 Readiness Probe**。没有 Readiness Probe 的 Pod 在启动后会立即加入 Service，如果应用初始化慢，前几个请求必然失败。另外，不要在 Readiness Probe 失败逻辑中包含 Pod 关闭的检测——Pod 关闭时 Kubernetes 会自动把它从 Endpoints 中移除，无需手动配合。

---

## 5. Ingress：统一外部流量入口

### 问题：LoadBalancer 太贵

如果你有 10 个微服务都需要对外暴露，创建 10 个 LoadBalancer Service 意味着 10 个云厂商负载均衡器——每个都有独立的 IP 和费用。

**Ingress** 是解决方案：**一个 Ingress 对应一个外部 IP，可以将多个域名/路径路由到不同的 Service**。

![图 5.9 — 一个 Ingress 路由多个服务](/book-notes/kubernetes-in-action/images/fig-5-9-ingress-multiple-services.png)

Ingress 工作在 HTTP 层（L7），可以基于 Host 头和 URL 路径做路由，这是 L4 的 Service 做不到的。

### Ingress Controller 是前提

Ingress 资源本身只是一个声明配置，需要 **Ingress Controller** 来真正处理流量。常见的实现：

- **Nginx Ingress Controller**（最常用，Minikube 内置）
- **GKE Ingress**（使用 Google Cloud HTTP Load Balancer）
- **Traefik**、**HAProxy**、**Istio Gateway** 等

不同云厂商和环境使用不同的 Controller 实现，但对外的 Ingress 资源格式是统一的。

### 多路径路由

把多个服务挂在同一域名的不同路径下：

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: kubia
spec:
  rules:
  - host: kubia.example.com
    http:
      paths:
      - path: /kubia
        backend:
          serviceName: kubia
          servicePort: 80
      - path: /foo
        backend:
          serviceName: bar
          servicePort: 80
```

### 多域名路由

把不同域名路由到不同服务：

```yaml
spec:
  rules:
  - host: foo.example.com
    http:
      paths:
      - path: /
        backend:
          serviceName: foo
          servicePort: 80
  - host: bar.example.com
    http:
      paths:
      - path: /
        backend:
          serviceName: bar
          servicePort: 80
```

### Ingress 的工作机制

关键细节：**Ingress Controller 不通过 Service 转发流量，而是直接转发到 Pod**。

流程如下（图 5.10）：

1. 客户端 DNS 查询 `kubia.example.com` → 返回 Ingress Controller 的 IP
2. 客户端发送 HTTP 请求，Host 头为 `kubia.example.com`
3. Ingress Controller 根据 Host/Path 查找对应的 Service
4. 通过该 Service 的 **Endpoints 对象**获取 Pod IP 列表
5. 直接将请求转发到某个 Pod

![图 5.10 — Ingress 访问 Pod 的完整流程](/book-notes/kubernetes-in-action/images/fig-5-10-ingress-accessing-pods.png)

Service 只用于"查找 Pod 列表"，实际流量不经过 Service 的 iptables 规则。

### TLS 终止

Ingress Controller 可以处理 HTTPS，把 TLS 证书存储在 Kubernetes Secret 中：

```yaml
spec:
  tls:
  - hosts:
    - kubia.example.com
    secretName: tls-secret   # 包含 tls.crt 和 tls.key
  rules:
  - host: kubia.example.com
    http:
      paths:
      - path: /
        backend:
          serviceName: kubia-nodeport
          servicePort: 80
```

TLS 在 Ingress Controller 层终止，后端 Pod 只处理 HTTP 流量，无需管理证书。

> [!tip] AI 平台场景
> 在 AI 平台中，Ingress 是统一入口的理想选择：
> - `api.example.com/v1/completions` → LLM Service（vLLM 集群）
> - `api.example.com/v1/embeddings` → Embedding Service（text-embeddings-inference）
> - `api.example.com/v1/images` → Image Generation Service（Stable Diffusion）
>
> 所有服务共享一个外部 IP 和 TLS 证书，用 Ingress 的路径规则分发。这比为每个服务创建独立 LoadBalancer 节省大量成本，也让 API 对外有统一的域名。

---

## 6. Headless Service：直接发现 Pod IP

### 普通 Service 的局限

普通 Service 通过 kube-proxy 做负载均衡，客户端只能看到一个虚拟 IP，无法知道后面有哪些具体的 Pod。大多数场景这是好事，但有些特殊场景需要**直接连接到特定 Pod**：

- **分布式训练**（PyTorch DDP、Horovod）：Worker 之间需要点对点通信，不能经过代理
- **有状态集群**（Cassandra、Elasticsearch）：客户端需要知道哪个节点有哪些数据，才能做本地路由
- **StatefulSet**：每个 Pod 有固定标识，客户端需要连接特定 Pod

### Headless Service 的工作方式

将 `clusterIP: None` 设置在 Service 规范中，Service 就变为 Headless：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-headless
spec:
  clusterIP: None    # 关键：不分配虚拟 IP
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

Headless Service 没有 ClusterIP，对它做 DNS 查询时，kube-dns 不返回单个 IP，而是返回**所有匹配 Pod 的 A 记录**：

```
$ nslookup kubia-headless
Name:    kubia-headless.default.svc.cluster.local
Address: 10.108.1.4     ← Pod A 的真实 IP
Name:    kubia-headless.default.svc.cluster.local
Address: 10.108.2.5     ← Pod B 的真实 IP
```

客户端通过 DNS 查询拿到所有 Pod IP，自行决定连接哪个。本质上是把负载均衡的控制权从 kube-proxy 转移到了客户端。

> [!note]
> Headless Service 虽然没有 ClusterIP，但仍然通过 DNS 提供了某种负载均衡——DNS 轮询（round-robin）。这与普通 Service 的区别在于：普通 Service 在内核态做 NAT，Headless 通过 DNS 让客户端自行选择。

---

## 7. 交互式演示

以下可视化展示三种 Service 类型的流量路径，点击顶部按钮切换场景：

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/service-traffic-flow.html" height="480px" title="Service 流量路由演示" />

---

## 总结

| 概念 | 核心作用 | 关键机制 |
|------|---------|---------|
| **Service** | 为一组 Pod 提供稳定的虚拟 IP 和 DNS | Label Selector + Endpoints + kube-proxy iptables |
| **ClusterIP** | 集群内部访问入口 | 虚拟 IP，只在集群内有意义 |
| **NodePort** | 通过节点端口暴露到外部 | 每个节点开放固定端口（30000-32767） |
| **LoadBalancer** | 云厂商外部负载均衡 | 自动申请云 LB，前置于 NodePort |
| **ExternalName** | 封装外部服务 DNS 别名 | 纯 DNS CNAME，不涉及代理 |
| **Readiness Probe** | 控制 Pod 何时接收流量 | 失败则从 Endpoints 移除，不重启容器 |
| **Ingress** | 统一 HTTP 入口，L7 路由 | Controller 直接转发到 Pod，不经过 Service iptables |
| **Headless Service** | 直接暴露 Pod IP 供客户端发现 | `clusterIP: None`，DNS 返回多个 A 记录 |

Service 是 Kubernetes 网络模型的基础抽象。理解它的工作机制——尤其是 Endpoints 对象和 kube-proxy 的 iptables 规则——是排查网络故障、设计高可用系统的前提。书中第 11 章会深入讲解 kube-proxy 的底层实现，到时会有更清晰的全貌。
