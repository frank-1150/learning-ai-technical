---
date: "2026-04-10"
title: "Kubernetes 安全体系：认证、授权与隔离"
description: "深入 RBAC 权限模型、Pod 安全上下文与网络策略，构建多租户 K8s 集群的安全基础"
tags: [kubernetes, rbac, security, serviceaccount, networkpolicy, podsecuritypolicy]
---

# Kubernetes 安全体系：认证、授权与隔离

> 本文对应原书 **第 12-13 章**，覆盖：**Securing the Kubernetes API server**（Ch12）与 **Securing cluster nodes and the network**（Ch13）。两章构成 Kubernetes 安全体系的完整闭环：从"谁能操作 API"到"容器能做什么"，再到"Pod 之间能互通吗"。

---

## Ch12：API Server 安全

### 认证（Authentication）：请求进入 API Server 的第一关

每个进入 Kubernetes API Server 的请求，首先要经过一组**认证插件（Authentication Plugins）**的审查。这些插件依次尝试从请求中提取客户端身份——谁发的这个请求？第一个能识别出来的插件会返回该客户端的 username、user ID 和所属 groups，API Server 随即跳过剩余插件，进入授权阶段。

认证插件支持多种方式识别身份：

- **客户端证书（Client Certificate）**：客户端持有由集群 CA 签发的 TLS 证书，证书的 Common Name 即为 username
- **Bearer Token**：放在 HTTP header `Authorization: Bearer <token>` 中，ServiceAccount 使用的就是这种
- **Basic HTTP 认证**：用户名 + 密码（不推荐生产使用）
- **OpenID Connect（OIDC）**：对接外部 Identity Provider（如 Google、Okta），适合企业 SSO 场景

Kubernetes 区分两类客户端：**人类用户**（通过外部 IdP 管理，K8s 本身不存储用户对象）和 **Pod 内运行的应用**（通过 ServiceAccount 认证）。这是一个重要的设计决策：K8s 不是一个 Identity Provider，它把人类用户的管理完全委托给外部系统。

**内置 Groups 的特殊含义**：

| Group | 含义 |
|-------|------|
| `system:unauthenticated` | 任何认证插件都无法识别的请求 |
| `system:authenticated` | 成功认证的所有用户 |
| `system:serviceaccounts` | 集群内所有 ServiceAccount |
| `system:serviceaccounts:<namespace>` | 特定 namespace 的所有 ServiceAccount |

---

### ServiceAccount：Pod 的"身份证"

每个 Pod 都关联一个 ServiceAccount——这是 Pod 向 API Server 证明自己身份的唯一凭证。ServiceAccount 本身是 K8s 资源对象（类似 Pod、ConfigMap），被限定在单个 namespace 内。

![图 12.1 每个 Pod 关联一个 ServiceAccount，且只能使用同 namespace 内的 ServiceAccount](/book-notes/kubernetes-in-action/images/fig-12-1-serviceaccount-namespace.png)

**ServiceAccount 的工作机制**：

每个 ServiceAccount 都关联一个 JWT token，被自动挂载到 Pod 的文件系统路径 `/var/run/secrets/kubernetes.io/serviceaccount/token`。Pod 内应用通过在 HTTP 请求的 `Authorization` header 中携带该 token 向 API Server 认证。API Server 校验 token 后，将 ServiceAccount 的 username（格式为 `system:serviceaccount:<namespace>:<name>`）传递给授权插件。

**每个 namespace 都有一个 `default` ServiceAccount**，它是在 namespace 创建时自动生成的。如果没有显式指定，Pod 会使用这个默认账号。

**创建自定义 ServiceAccount** 非常简单：

```bash
kubectl create serviceaccount my-app-sa
```

可以在 Pod 的 `spec.serviceAccountName` 字段中指定：

```yaml
spec:
  serviceAccountName: my-app-sa
  containers:
  - name: main
    image: my-app:latest
```

> [!note] ServiceAccount 必须在创建 Pod 时指定，不能后期修改。

**Mountable Secrets 限制**：ServiceAccount 可以通过注解 `kubernetes.io/enforce-mountable-secrets=true` 限制 Pod 只能挂载该 ServiceAccount 白名单内的 Secret，防止 Pod 越权访问敏感配置。

**Image Pull Secrets 集成**：将 registry 凭证放入 ServiceAccount 的 `imagePullSecrets` 列表后，该 SA 的所有 Pod 会自动获得拉取镜像的权限，无需在每个 Pod 中重复声明。

---

### RBAC：基于角色的访问控制

**为什么需要 RBAC？**

在 K8s 1.6 以前，集群安全形同虚设——任何人拿到 ServiceAccount token 就能对集群为所欲为。通过路径遍历漏洞（Directory Traversal）获取 token，然后横向渗透整个集群，这类攻击案例在早年屡见不鲜。从 1.8 版本起，RBAC 升为 GA 并在大多数集群中默认开启，从根本上改变了这一局面。

**RBAC 的核心思想是最小权限原则（Principle of Least Privilege）**：每个主体只拥有完成其职责所必需的最小权限集合，不多给一分。

**HTTP 方法到 RBAC Verb 的映射**（Table 12.1）：

| HTTP 方法 | 单个资源 verb | 集合操作 verb |
|-----------|-------------|-------------|
| GET, HEAD | get（watch） | list（watch） |
| POST | create | — |
| PUT | update | — |
| PATCH | patch | — |
| DELETE | delete | deletecollection |

此外，`use` verb 专用于 PodSecurityPolicy 资源，非资源 URL（如 `/healthz`）则使用小写的 HTTP 方法名（`get`、`post` 等）。

---

#### 四个核心 RBAC 资源

RBAC 的整个权限体系由四种资源构建：

![图 12.2 Role 定义"能做什么"，RoleBinding 定义"谁能做"](/book-notes/kubernetes-in-action/images/fig-12-2-rbac-rolebinding.png)

**1. Role（命名空间级别）**

Role 定义在某个 namespace 内"能对哪些资源执行哪些操作"。下面是一个允许读取 Services 的 Role：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: foo
  name: service-reader
rules:
- apiGroups: [""]          # core API group，Services 属于此 group
  verbs: ["get", "list"]
  resources: ["services"]  # 注意：必须用复数形式
```

一个 Role 可以包含多条规则（rules），每条规则指定 apiGroups、resources 和允许的 verbs。若需要访问多个 API group 的资源，需要写多条规则。

**2. ClusterRole（集群级别）**

ClusterRole 不属于任何 namespace，它的使用场景有三类：

- 授权访问**集群级别资源**（Nodes、PersistentVolumes、Namespaces 等）
- 授权访问**非资源 URL**（`/healthz`、`/api`、`/metrics`）
- 作为**通用模板**被各 namespace 内的 RoleBinding 复用，避免在每个 namespace 中重复定义相同的 Role

```bash
# 创建允许 list/get PersistentVolumes 的 ClusterRole
kubectl create clusterrole pv-reader --verb=get,list --resource=persistentvolumes
```

**3. RoleBinding（命名空间级别绑定）**

RoleBinding 把 Role（或 ClusterRole）绑定到一个或多个 Subjects（User、ServiceAccount、Group），作用范围限定在 RoleBinding 所在的 namespace。

```bash
# 把 service-reader Role 绑定给 foo namespace 的 default ServiceAccount
kubectl create rolebinding test --role=service-reader \
  --serviceaccount=foo:default -n foo
```

生成的 YAML 结构：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: test
  namespace: foo
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: service-reader
subjects:
- kind: ServiceAccount
  name: default
  namespace: foo
```

**关键点**：RoleBinding 的 `subjects` 可以包含来自不同 namespace 的 ServiceAccount。这意味着你可以把 `bar` namespace 的 ServiceAccount 也加入 `foo` namespace 的 RoleBinding，让它同样能 list `foo` namespace 的 Services。

**4. ClusterRoleBinding（集群级别绑定）**

ClusterRoleBinding 绑定 ClusterRole，权限生效范围是**全集群所有 namespace**。

```bash
kubectl create clusterrolebinding pv-test --clusterrole=pv-reader \
  --serviceaccount=foo:default
```

> [!warning] RoleBinding + ClusterRole 的组合陷阱
> 用 RoleBinding（非 ClusterRoleBinding）引用一个 ClusterRole，权限只在 RoleBinding 所在的 namespace 生效，**不能**访问集群级别资源（如 PersistentVolumes）。要访问集群级别资源，必须用 ClusterRoleBinding。

**何时用什么组合**（Table 12.2）：

| 访问目标 | Role 类型 | Binding 类型 |
|---------|----------|-------------|
| 集群级别资源（Nodes、PV...） | ClusterRole | ClusterRoleBinding |
| 非资源 URL（/api、/healthz...） | ClusterRole | ClusterRoleBinding |
| 所有 namespace 的命名空间资源 | ClusterRole | ClusterRoleBinding |
| 特定 namespace 资源（复用 ClusterRole） | ClusterRole | RoleBinding |
| 特定 namespace 资源（各自定义 Role） | Role | RoleBinding |

---

#### 内置默认 ClusterRole

Kubernetes 开箱即用地提供了几个重要的 ClusterRole：

| ClusterRole | 权限范围 |
|------------|---------|
| `view` | 只读访问大多数命名空间资源（不含 Secrets，防止权限提升） |
| `edit` | 读写命名空间资源（不含 Roles/RoleBindings，防止权限提升） |
| `admin` | 命名空间的完整控制（不含 ResourceQuota 和 Namespace 本身） |
| `cluster-admin` | 整个集群的完整控制 |

以 `system:` 前缀开头的 ClusterRole 是系统组件（kube-scheduler、kubelet、kube-dns 等）使用的，不应手动修改。

> [!tip] 权限提升防护
> API Server 只允许用户创建或更新 Role/ClusterRole，当且仅当该用户自身已拥有该 Role 内包含的所有权限。这防止了通过创建一个包含超出自身权限的 Role 来实现权限提升。

---

#### 交互式 RBAC 权限模型

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/rbac-model.html" height="500px" title="RBAC 权限模型交互图" />

---

### ServiceAccount 最佳实践

**不要在所有 Pod 上共用 `default` ServiceAccount**。这是最常见的安全反模式。`default` SA 默认只有极少权限，但一旦通过 RoleBinding 给它授权，namespace 内所有 Pod 都会获得该权限——包括那些根本不需要这个权限的 Pod。

**为每个应用创建专用 ServiceAccount**，按最小权限原则单独授权：

```bash
kubectl create serviceaccount monitoring-sa -n monitoring
kubectl create rolebinding monitoring-binding \
  --clusterrole=view \
  --serviceaccount=monitoring:monitoring-sa \
  -n monitoring
```

**自动挂载 token 的隐患**：默认情况下，Kubernetes 会把 ServiceAccount token 自动挂载到每个容器。如果 Pod 不需要访问 API Server（大多数业务 Pod 都不需要），可以在 ServiceAccount 或 Pod spec 中禁用：

```yaml
spec:
  automountServiceAccountToken: false
```

**设计原则**：始终假设你的 Pod 可能被攻陷，通过 RBAC 限制被攻陷后的爆炸半径（blast radius）。攻击者拿到 token，能做的只有 RBAC 允许的操作——不多不少。

**在 AI 平台中的应用**：多团队共用同一个 K8s 集群时，每个团队的模型训练/推理 Pod 应使用各自专属的 ServiceAccount，并通过 RoleBinding 限制它们只能访问自己 namespace 内的资源——读不到其他团队的模型权重，也改不了其他团队的 Deployment。

---

## Ch13：节点与网络安全

### 宿主机命名空间：高风险特权配置

在继续讨论容器安全上下文之前，有必要理解 Pod 与宿主机 Linux 命名空间的关系。

默认情况下，每个 Pod 有独立的网络命名空间（独立 IP 和端口空间）、PID 命名空间（独立进程树）和 IPC 命名空间。这三个配置可以通过 Pod spec 绕过，使 Pod 直接使用宿主机的命名空间：

```yaml
spec:
  hostNetwork: true  # 使用宿主机网络接口，Pod 没有独立 IP
  hostPID: true      # 可以看到宿主机上所有进程
  hostIPC: true      # 可以通过 IPC 与宿主机进程通信
```

**何时合理使用**：系统级 DaemonSet（如 kube-proxy、节点监控代理）通常需要访问宿主机网络或 PID 命名空间。常规业务 Pod 绝不应使用这些选项——一旦容器被攻陷，攻击者可以直接操作宿主机网络或 kill 宿主机进程。

**hostPort 与 NodePort 的区别**：`hostPort` 让容器绑定到所在节点的特定端口，但这个绑定是"独占"的——同一节点上最多只能跑一个使用该 hostPort 的 Pod。而 NodePort 服务在每个节点上都开放端口，并通过 iptables 将流量转发到任意 Pod。

---

### Pod SecurityContext：细粒度容器权限控制

`securityContext` 是 Pod 和容器 spec 中的关键安全配置项。它允许在不授予 `privileged: true` 全权限的前提下，精细控制容器的权限边界。

#### runAsUser / runAsNonRoot：以非 root 身份运行

默认情况下，如果容器镜像没有指定 `USER` 指令，容器以 **root（UID 0）** 运行。这在容器环境中仍然是危险的——当挂载了宿主机目录或访问宿主机资源时，root 权限会造成真实损害。

```yaml
spec:
  containers:
  - name: main
    securityContext:
      runAsUser: 405       # 指定 UID
      runAsNonRoot: true   # 禁止以 root 运行，镜像内置 root 用户会被拒绝
```

`runAsNonRoot: true` 是一个防御层：即使攻击者替换了容器镜像并配置为以 root 运行，K8s 也会拒绝调度该 Pod（Pod 会被调度但无法启动，status 显示 `container has runAsNonRoot and image will run as root`）。

#### privileged：等同于宿主机 root 的"核武器"

```yaml
securityContext:
  privileged: true
```

特权容器可以看到宿主机上所有的设备文件（`/dev` 目录），可以任意修改宿主机内核参数，可以加载/卸载内核模块——**这等同于在宿主机上以 root 直接运行代码**。

典型合法用例：`kube-proxy` 需要修改宿主机的 `iptables` 规则；GPU 驱动 DaemonSet 需要访问 `/dev/nvidia*` 设备。业务 Pod 不应设置此选项。

#### readOnlyRootFilesystem：只读容器文件系统

```yaml
securityContext:
  readOnlyRootFilesystem: true
```

这一配置将容器根文件系统设为只读。即使攻击者利用代码漏洞在容器内获得代码执行能力，也无法修改应用文件（如替换 PHP 文件、写入 cron job），大幅限制了持久化攻击的可能。

实践上，需要配合 volumeMount 为需要写入的目录（日志目录、临时文件等）挂载可写卷：

```yaml
spec:
  containers:
  - name: main
    securityContext:
      readOnlyRootFilesystem: true
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: logs
    emptyDir: {}
  - name: tmp
    emptyDir: {}
```

#### Linux Capabilities：细粒度内核权限

Linux kernel 把 root 的全权限拆分成约 40 个独立的 capability（能力），每个 capability 控制一类内核操作。Kubernetes 允许在不开启 privileged 的情况下，针对单个容器增加或删除 capability：

```yaml
securityContext:
  capabilities:
    add:
    - SYS_TIME     # 允许修改系统时钟
    drop:
    - CHOWN        # 禁止修改文件所有权（默认容器有此权限）
    - NET_RAW      # 禁止发送原始 IP 报文（防止 ARP 欺骗等攻击）
```

**常用 capability 说明**：

| Capability | 作用 | 典型需求方 |
|-----------|------|-----------|
| `NET_ADMIN` | 修改网络配置、iptables | CNI 插件、网络代理 |
| `SYS_ADMIN` | 大量系统管理操作 | 高度危险，应避免 |
| `SYS_TIME` | 修改系统时钟 | NTP 同步服务 |
| `CHOWN` | 修改文件所有权 | 默认开启，多数情况可 drop |
| `SYS_MODULE` | 加载/卸载内核模块 | 驱动安装，高度危险 |

> [!tip] 安全加固建议
> 生产环境建议采用"drop all + add back only what's needed"策略：先 `drop: ["ALL"]`，再按需 `add` 必要的 capability。这比默认配置更安全，也比 `privileged: true` 的爆炸半径小得多。

#### Pod 级别 SecurityContext

除了容器级别，`securityContext` 也可以在 `spec.securityContext`（Pod 级别）设置，作为所有容器的默认值：

```yaml
spec:
  securityContext:
    fsGroup: 555           # 挂载卷的 group owner 设为 555
    supplementalGroups: [666, 777]  # 容器用户的附加 group
  containers:
  - name: first
    securityContext:
      runAsUser: 1111      # 容器级别覆盖 Pod 级别
```

`fsGroup` 解决了多容器共享 Volume 时的权限问题：以不同 UID 运行的两个容器可能都无法读写对方创建的文件，但通过 `fsGroup` 设置一个共同的 group，让 Volume 目录归属该 group，两个容器都加入这个 group，问题得以解决。

---

### PodSecurityPolicy（PSP）：集群级别的安全基准线

PodSecurityPolicy 是一个集群级别（non-namespaced）的资源，定义了用户可以在 Pod 中使用哪些安全配置。当用户提交 Pod 时，PSP Admission Controller 会对比所有可用策略，若 Pod 违反任意被激活策略，直接拒绝创建。

> [!warning] PSP 已在 K8s 1.21 废弃，1.25 正式移除
> 本节作为概念理解仍有价值，现代替代方案是 **Pod Security Admission（PSA）**（内置）或 **OPA Gatekeeper / Kyverno**（外部 webhook）。

一个典型的 PSP 配置示例：

```yaml
apiVersion: extensions/v1beta1
kind: PodSecurityPolicy
metadata:
  name: default
spec:
  hostIPC: false
  hostPID: false
  hostNetwork: false
  privileged: false               # 禁止 privileged 容器
  readOnlyRootFilesystem: true    # 强制只读根文件系统
  runAsUser:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  supplementalGroups:
    rule: RunAsAny
  volumes:
  - '*'
```

PSP 与 RBAC 配合使用——通过 ClusterRole 的 `use` verb 控制哪个用户/ServiceAccount 可以使用哪个 PSP：

```bash
# 允许所有认证用户使用 default PSP（限制性策略）
kubectl create clusterrole psp-default --verb=use \
  --resource=podsecuritypolicies --resource-name=default
kubectl create clusterrolebinding psp-all-users \
  --clusterrole=psp-default --group=system:authenticated

# 只允许 bob 使用 privileged PSP
kubectl create clusterrolebinding psp-bob \
  --clusterrole=psp-privileged --user=bob
```

**现代替代方案**：

- **Pod Security Admission（K8s 1.22+）**：内置三级策略（Privileged / Baseline / Restricted），通过 namespace 标签激活，简单但功能有限
- **OPA Gatekeeper**：基于 Rego 策略语言，灵活且可审计，社区主流选择
- **Kyverno**：YAML 原生策略语言，上手门槛更低

---

### NetworkPolicy：Pod 间网络隔离

#### 为什么需要 NetworkPolicy？

Kubernetes 的默认网络模型是**全通（flat networking）**：同一集群内所有 Pod 默认可以互相访问，不受命名空间限制。这在横向移动（Lateral Movement）攻击场景下极为危险——一旦攻击者控制了某个 Pod，可以直接尝试访问集群内所有其他服务。

NetworkPolicy 通过 Ingress（入站）和 Egress（出站）规则，定义哪些 Pod 之间可以通信，哪些不行。

> [!note] NetworkPolicy 依赖 CNI 插件支持
> 不是所有 CNI 插件都支持 NetworkPolicy。Calico、Cilium、Weave Net、Antrea 支持；Flannel 默认不支持。在使用前需确认集群的网络插件。

#### Default Deny：先关门再开白名单

最佳实践是先为每个 namespace 创建一个"默认拒绝所有入站"的 NetworkPolicy，再逐条开放需要的访问：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
spec:
  podSelector: {}   # 空选择器 = 匹配 namespace 内所有 Pod
  # 没有 ingress 规则 = 拒绝所有入站流量
```

#### 按 Pod 标签开放访问

为数据库 Pod 开放来自 webserver Pod 的访问：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: postgres-netpolicy
spec:
  podSelector:
    matchLabels:
      app: database       # 此策略保护的 Pod
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: webserver  # 只允许此标签的 Pod 进来
    ports:
    - port: 5432          # 只允许此端口
```

![图 13.4 NetworkPolicy 允许 webserver Pod 访问 database Pod 的 5432 端口，其他 Pod 全部拒绝](/book-notes/kubernetes-in-action/images/fig-13-4-networkpolicy-postgres.png)

#### 按 Namespace 标签隔离多租户

在多租户场景中（多个团队共用同一集群），可以基于 namespace 标签来隔离流量，只允许同一租户的 namespace 互访：

```yaml
spec:
  podSelector:
    matchLabels:
      app: shopping-cart
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          tenant: manning  # 只允许标记为 tenant=manning 的 namespace 内的 Pod
    ports:
    - port: 80
```

#### Egress 规则：限制出站流量

不仅可以控制入站，也可以限制 Pod 的出站目标：

```yaml
spec:
  podSelector:
    matchLabels:
      app: webserver
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database     # webserver 只能连接 database Pod
```

**CIDR 块**：Ingress/Egress 规则还支持 IP 块限定：

```yaml
ingress:
- from:
  - ipBlock:
      cidr: 192.168.1.0/24    # 只允许此 IP 段访问
```

#### NetworkPolicy 在 AI 平台中的应用

在 GPU 推理集群中，NetworkPolicy 能有效控制服务间的访问边界：

- **推理服务**：只接受来自 API Gateway namespace 的流量，拒绝直接访问
- **模型存储（对象存储 Pod）**：只允许模型加载 Job 访问，推理 Pod 只能通过模型加载器间接获取
- **向量数据库**：只允许 RAG 检索服务访问，其他 Pod 均拒绝
- **监控 Prometheus**：使用 namespaceSelector 允许从 monitoring namespace 采集所有业务 Pod 的 metrics

---

## 总结：Kubernetes 安全分层模型

从 Ch12-13 的内容可以抽象出 Kubernetes 安全的三道防线：

| 防线 | 机制 | 防御的威胁 |
|------|------|-----------|
| **认证层** | ServiceAccount + JWT Token + TLS 证书 | 身份冒充、未授权 API 访问 |
| **授权层** | RBAC（Role/ClusterRole + Binding） | 权限滥用、横向权限提升 |
| **隔离层** | SecurityContext + PSP/PSA + NetworkPolicy | 容器逃逸、横向移动 |

**最小权限原则贯穿始终**：

1. 为每个 Pod 创建专用 ServiceAccount，按需授权 RBAC
2. 禁用不需要的 ServiceAccount token 自动挂载
3. 容器以非 root 用户运行（`runAsNonRoot: true`）
4. 开启只读根文件系统（`readOnlyRootFilesystem: true`）
5. Drop 所有默认 capabilities，按需 add
6. 用 NetworkPolicy default-deny 隔离命名空间，再开放必要路径

安全不是一道门，而是一系列纵深防御层。每一层独立失效时，下一层仍能限制损害范围——这是分布式系统安全设计的基本思路。
