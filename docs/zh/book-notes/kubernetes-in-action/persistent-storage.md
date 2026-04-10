---
date: 2026-04-10
title: "持久化存储：模型权重与有状态数据的管理之道"
description: "从 emptyDir 到 PersistentVolume，理解 Kubernetes 存储抽象层次及其在 AI 工作负载中的应用"
tags: [kubernetes, storage, pvc, pv, storageclass, volumes]
---

# 持久化存储：模型权重与有状态数据的管理之道

> 本文对应原书 **第 6 章**，覆盖：Volumes: attaching disk storage to containers

Pod 和容器让我们能以声明式的方式部署无状态服务，但现实中几乎所有有价值的应用都需要存储数据——数据库写入记录、训练任务保存 checkpoint、推理服务加载模型权重。Kubernetes 的存储体系从临时的 `emptyDir` 到弹性扩展的 StorageClass，构建了一套完整的抽象层次。理解这套抽象，是运营任何有状态 AI 工作负载的基础。

## 一、容器存储的本质问题

### 为什么容器不能依赖自身文件系统

容器的文件系统来自镜像。每次容器启动时，它拿到的是镜像构建时确定好的一份"初始状态"——只读层加上一个空白的可写层（Union Mount 的顶层）。这意味着：

- 容器重启后，可写层被销毁，写入的任何数据都消失了
- 同一 Pod 中的多个容器各自拥有隔离的文件系统，无法直接共享数据
- 即便 Pod 还在运行，Liveness Probe 失败触发容器重建时，之前写入的内容也会丢失

对于 Web 服务，这通常没问题——无状态服务本来就不该在本地磁盘写持久数据。但对于以下场景，这是个致命缺陷：

- **日志聚合 sidecar**：主容器写日志，日志收集容器需要读同一目录
- **模型推理服务**：启动时需要从磁盘加载几十 GB 的模型权重文件
- **数据库 Pod**：写入的数据必须在 Pod 重建后依然存在
- **训练 checkpoint**：长时间训练任务中途需要保存进度

### Volume 的设计目标

Kubernetes 通过 **Volume（卷）** 解决这个问题。Volume 不是独立的 Kubernetes 资源，而是 Pod spec 的一部分，与 Pod 共享生命周期。关键特性：

1. **共享性**：同一 Pod 的多个容器可以挂载同一个 Volume，实现数据共享
2. **持久性（相对）**：容器重启不会破坏 Volume 的内容，Pod 重建才会（对于某些类型而言）
3. **声明式挂载**：在 Pod spec 中声明 Volume，并在每个需要访问它的容器中声明 `VolumeMount`——未声明 VolumeMount 的容器即便在同一 Pod 中也无法访问该 Volume

下图（原书 Figure 6.2）展示了三个容器通过两个 Volume 实现协作的经典架构：WebServer 和 ContentAgent 共享 `publicHtml` volume，WebServer 和 LogRotator 共享 `logVol` volume。每个容器只需声明自己需要的 VolumeMount，未声明的容器即便在同一 Pod 中也无法访问。

---

## 二、Volume 类型全览

Kubernetes 支持大量不同类型的 Volume，从临时存储到跨节点的网络存储都有覆盖。

### `emptyDir`：Pod 级别临时存储

`emptyDir` 是最简单的 Volume 类型。顾名思义，它在 Pod 启动时创建一个空目录，Pod 销毁时一起删除。

```yaml
volumes:
  - name: html
    emptyDir: {}
```

**适用场景：**
- 同一 Pod 中多个容器之间共享临时文件
- 大数据集的磁盘排序（数据不适合放内存时的临时缓冲区）
- 容器需要写临时文件，但镜像的文件系统是只读的

**性能调优**：默认 `emptyDir` 存储在节点的磁盘上。如果需要内存级速度（比如高频读写的临时缓存），可以让它使用 `tmpfs`：

```yaml
volumes:
  - name: cache
    emptyDir:
      medium: Memory
```

**在 AI 场景中**：推理服务的 KV Cache（键值缓存）是一个典型的 `emptyDir` 使用场景。每个推理 Pod 在请求处理过程中需要大量临时缓冲区，这些数据不需要跨 Pod 存活，`emptyDir` 是完美选择。

### `hostPath`：宿主机目录挂载

`hostPath` 将节点的某个目录直接挂载进容器，是第一种"真正持久"的 Volume——Pod 删除后，数据依然留在节点文件系统上。

```yaml
volumes:
  - name: varlog
    hostPath:
      path: /var/log
```

**为什么危险（Why 不要轻易使用）**：

`hostPath` 破坏了 Pod 的**可移植性**。Pod 与特定节点绑定，重调度到另一节点后看到的是不同的数据，会导致数据不一致。同时，挂载宿主机目录带来安全风险，恶意容器可能借此逃逸到节点文件系统。

**合理使用场景**：系统级别的 DaemonSet Pod（如日志收集器 `fluentd`、监控 agent `node-exporter`）需要读取节点的 `/var/log`、`/proc` 等目录，这是 `hostPath` 的正当用途。

```bash
# 查看 fluentd 使用的 hostPath volumes
$ kubectl describe po fluentd-kubia-xxxx --namespace kube-system
Volumes:
  varlog:
    Type:    HostPath
    Path:    /var/log
  varlibdockercontainers:
    Type:    HostPath
    Path:    /var/lib/docker/containers
```

**原则**：`hostPath` 只用于读写节点系统文件，绝不用于跨 Pod 持久化业务数据。

### `gitRepo`：Git 仓库内容初始化

`gitRepo` 本质上是一个 `emptyDir`，Pod 启动时会自动 clone 指定的 Git 仓库内容到目录中。Pod 运行后，Volume 内容不会再与仓库同步（静态快照）。

```yaml
volumes:
  - name: html
    gitRepo:
      repository: https://github.com/luksa/kubia-website-example.git
      revision: master
      directory: .
```

**注意**：`gitRepo` 不支持私有仓库（无法配置 SSH key）。需要持续同步的场景，应使用 git-sync sidecar 容器。

### 云存储：跨节点真正的持久化

当应用需要在 Pod 重建或跨节点调度后依然保留数据时，必须使用网络附加存储（NAS）。Kubernetes 原生支持主流云提供商的块存储：

| Volume 类型 | 云平台 | 特点 |
|---|---|---|
| `gcePersistentDisk` | Google Cloud | GCE Persistent Disk，需与集群同 zone |
| `awsElasticBlockStore` | AWS | EBS Volume，需与 EC2 节点同 AZ |
| `azureDisk` | Azure | Azure Managed Disk |
| `nfs` | 自建机房 | NFS 共享，需指定 server IP 和 path |

直接使用云存储 Volume 的示例（GCE PD）：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mongodb
spec:
  volumes:
    - name: mongodb-data
      gcePersistentDisk:
        pdName: mongodb        # 须先用 gcloud 创建
        fsType: ext4
  containers:
    - image: mongo
      name: mongodb
      volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
```

**核心问题**：这个 Pod manifest 把 GCE 的具体信息（`gcePersistentDisk`、`pdName`）硬编码进去了。这份 YAML 无法在 AWS 集群上运行，破坏了可移植性——这是 PersistentVolume 抽象出现的根本原因。

**在 AI 场景中**：模型权重文件（几十 GB 到几百 GB）通常存储在 NFS 或云对象存储挂载点上。多个推理 Pod 需要同时读取同一份模型权重，这要求存储支持多节点并发只读访问，NFS 或 `ReadOnlyMany` 模式的云存储是标准方案。

---

## 三、PersistentVolume 与 PersistentVolumeClaim：解耦开发者与基础设施

### 为什么需要这层抽象

直接在 Pod spec 中引用云存储有一个根本性的设计问题：**它要求应用开发者了解底层基础设施的细节**。

Kubernetes 的核心设计理念之一是：应用开发者只应关心"我需要什么"，而不是"基础设施如何提供"。开发者创建 Pod 时，不需要知道底层跑的是什么 CPU；同理，也不应该需要知道底层用的是 GCE PD 还是 AWS EBS。

PersistentVolume（PV）和 PersistentVolumeClaim（PVC）通过**角色分离**解决了这个问题：

- **集群管理员（Admin）**：负责了解底层基础设施，创建 PersistentVolume 资源，描述可用的存储容量和访问模式
- **应用开发者（User）**：不需要了解底层，只创建 PersistentVolumeClaim，声明"我需要多少存储、什么访问模式"

原书 Figure 6.6 清晰展示了这个流程（下方交互式可视化中的"存储抽象层次"标签页也有完整图示）：

完整流程如下：
1. Admin 在基础设施层创建网络存储（NFS export 或云存储磁盘）
2. Admin 向 Kubernetes API 提交 PersistentVolume manifest，描述这块存储
3. User（开发者）创建 PersistentVolumeClaim，声明所需的存储大小和访问模式
4. Kubernetes 自动找到匹配条件的 PV，将 PVC 绑定到该 PV
5. User 创建 Pod，在 volumes 中引用该 PVC 的名字

### 创建 PersistentVolume（管理员视角）

PV 是**集群级别的资源**（不属于任何 Namespace），由集群管理员创建：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mongodb-pv
spec:
  capacity:
    storage: 1Gi                    # 这块 PV 的存储容量
  accessModes:
    - ReadWriteOnce                 # 单节点读写
    - ReadOnlyMany                  # 多节点只读
  persistentVolumeReclaimPolicy: Retain  # PVC 删除后保留数据
  gcePersistentDisk:                # 底层实际存储类型（对开发者透明）
    pdName: mongodb
    fsType: ext4
```

关键字段解释：
- `capacity.storage`：PV 的容量声明，PVC 绑定时必须满足
- `accessModes`：该 PV 支持的访问模式（下一节详解）
- `persistentVolumeReclaimPolicy`：PVC 被删除后 PV 的处理策略

创建后查看状态：
```bash
$ kubectl get pv
NAME         CAPACITY   ACCESSMODES   STATUS      CLAIM
mongodb-pv   1Gi        RWO,ROX       Available
```

`Available` 表示 PV 已就绪，等待被 PVC 绑定。

### 创建 PersistentVolumeClaim（开发者视角）

PVC 是**Namespace 级别的资源**，由应用开发者创建。注意 PVC 对底层存储一无所知：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mongodb-pvc
spec:
  resources:
    requests:
      storage: 1Gi          # 需要至少 1Gi 的存储空间
  accessModes:
    - ReadWriteOnce         # 需要单节点读写
  storageClassName: ""      # 空字符串：强制绑定预创建的 PV（下一节解释）
```

Kubernetes 会自动找到满足条件的 PV（容量 >= 1Gi，访问模式包含 ReadWriteOnce），将其绑定：

```bash
$ kubectl get pvc
NAME          STATUS   VOLUME       CAPACITY   ACCESSMODES   AGE
mongodb-pvc   Bound    mongodb-pv   1Gi        RWO,ROX       3s
```

`Bound` 状态表示绑定成功。此时 `mongodb-pv` 的状态也变为 `Bound`。

### 在 Pod 中使用 PVC

Pod 引用 PVC 的名字，而不是 PV 的名字，也不是底层存储的任何细节：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mongodb
spec:
  containers:
    - image: mongo
      name: mongodb
      volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
  volumes:
    - name: mongodb-data
      persistentVolumeClaim:
        claimName: mongodb-pvc   # 引用 PVC 名字
```

这份 Pod manifest 完全与底层基础设施解耦。同样的 manifest 可以在 GKE、EKS、自建集群上运行——只要集群管理员提供了对应的 PV 即可。

### PV 的生命周期

PV 有四个状态：

```
Available → Bound → Released → (Recycled/Deleted/Retained)
```

| 状态 | 含义 |
|---|---|
| `Available` | PV 已创建，等待 PVC 绑定 |
| `Bound` | 已绑定到某个 PVC，该 PVC 独占此 PV |
| `Released` | PVC 已删除，PV 尚未被回收 |
| `Failed` | 自动回收失败 |

**重要**：PVC 删除后，PV 进入 `Released` 状态而非直接 `Available`——因为上一个 PVC 可能已经写入了数据，不应该在未清理的情况下直接被新 PVC 使用（可能导致数据泄露给不同的租户）。

回收策略（`persistentVolumeReclaimPolicy`）：
- `Retain`：保留 PV 和底层数据，需管理员手动处理后才能再用
- `Recycle`（已废弃）：清空数据后重新变为 Available
- `Delete`：自动删除底层存储资源（云存储磁盘）

---

## 四、StorageClass：动态供应，解放运维

### 为什么需要动态供应

PV/PVC 的抽象解决了可移植性问题，但仍有一个痛点：**管理员必须手动预创建 PV**。

在大规模集群中，这意味着：
- 需要预估并预创建大量不同规格的 PV
- PV 利用率低（分配了 50Gi 但实际只用了 2Gi）
- 运维工作量随集群规模线性增长
- AI 训练任务的存储需求往往是突发性的，很难提前预备

**StorageClass** 是解决方案：管理员不再预创建 PV，而是定义"存储配方"（StorageClass），让系统在 PVC 创建时自动按需生成 PV。

### StorageClass 定义

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: kubernetes.io/gce-pd    # 使用哪个 provisioner 来创建 PV
parameters:
  type: pd-ssd                        # 传递给 provisioner 的参数
  zone: europe-west1-b
```

StorageClass 同样是**集群级别的资源**（不属于任何 Namespace）。

`provisioner` 字段指定具体使用哪个插件来自动创建底层存储。Kubernetes 内置了主流云提供商的 provisioner（GCE PD、AWS EBS、Azure Disk 等），自建机房通常使用 NFS provisioner 或 Rook/Ceph。

### 动态供应的 PVC

在 PVC 中通过 `storageClassName` 引用 StorageClass：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mongodb-pvc
spec:
  storageClassName: fast           # 指定使用哪个 StorageClass
  resources:
    requests:
      storage: 100Mi
  accessModes:
    - ReadWriteOnce
```

PVC 创建后，Kubernetes 自动触发以下动作：
1. 找到名为 `fast` 的 StorageClass
2. 调用其 provisioner（`kubernetes.io/gce-pd`），按 PVC 的需求创建一块 GCE SSD 磁盘
3. 自动创建对应的 PV 资源
4. 将 PVC 绑定到新建的 PV

整个过程无需管理员干预。

原书 Figure 6.10 展示了动态供应的完整流程，下面的交互式可视化也有动画演示。

### 默认 StorageClass

集群通常会有一个默认 StorageClass（通过 annotation `storageclass.beta.kubernetes.io/is-default-class: "true"` 标记）：

```bash
$ kubectl get sc
NAME               TYPE
fast               kubernetes.io/gce-pd
standard (default) kubernetes.io/gce-pd
```

如果 PVC 不指定 `storageClassName`，默认 StorageClass 会自动介入，动态创建 PV。

**重要技巧**：如果你想让 PVC 绑定到手动预创建的 PV（而不触发动态供应），必须将 `storageClassName` 显式设为空字符串 `""`。省略这个字段和设为空字符串是两种不同行为：
- 省略：使用默认 StorageClass → 触发动态供应
- 设为 `""`：禁用动态供应 → 寻找匹配的预创建 PV

### 在 AI 推理集群中的应用

动态供应对 AI 工作负载尤其重要：

**模型权重的分发**：
- 预训练模型权重（如 70B 参数模型，约 140GB）通常存储在共享 NFS 或云存储上
- 每个推理 Pod 通过 PVC 挂载，使用 `ReadOnlyMany` 访问模式实现多 Pod 并发只读
- StorageClass 的动态供应确保新 Pod 随时可以获得存储，无需运维预备

**训练 checkpoint**：
- 分布式训练需要多个 worker 定期写入 checkpoint 到共享存储
- `ReadWriteMany` PVC 允许多个 Pod 同时写入（需要底层存储支持，如 NFS、Ceph）

---

## 五、访问模式对比

PV 和 PVC 的访问模式是绑定的核心匹配条件之一。注意：访问模式描述的是**节点级别**的访问约束，不是 Pod 级别：

| 访问模式 | 缩写 | 含义 |
|---|---|---|
| `ReadWriteOnce` | RWO | 只有一个节点可以同时以读写模式挂载 |
| `ReadOnlyMany` | ROX | 多个节点可以同时只读挂载 |
| `ReadWriteMany` | RWX | 多个节点可以同时读写挂载 |

**注意**：RWO 不等于"只有一个 Pod 可以使用"。同一节点上的多个 Pod 都可以挂载同一个 RWO 的 PVC；限制的是不同节点之间的并发挂载。

### 为什么模型权重存储应使用 ReadOnlyMany

推理集群中，多个 GPU 节点上的推理 Pod 需要同时读取模型权重。如果使用 `ReadWriteOnce`，只有一个节点能挂载，其他节点的 Pod 无法启动。

`ReadOnlyMany` 完美匹配这个场景：
1. 模型权重本来就是只读的，不需要写权限
2. 允许任意多个节点并发挂载，支持水平扩展推理 Pod
3. 从安全角度，只读挂载防止推理容器意外修改模型文件

```yaml
# 模型权重的 PVC（开发者视角）
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llm-model-weights
spec:
  storageClassName: nfs-fast      # NFS 支持 ReadOnlyMany
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 200Gi
```

```yaml
# 推理 Pod 挂载模型权重
spec:
  volumes:
    - name: model-weights
      persistentVolumeClaim:
        claimName: llm-model-weights
  containers:
    - name: vllm-inference
      image: vllm/vllm-openai:latest
      volumeMounts:
        - name: model-weights
          mountPath: /models
          readOnly: true
```

---

## 六、交互式可视化

<HtmlVisualization src="/book-notes/kubernetes-in-action/visualizations/storage-abstraction-layers.html" height="500px" title="K8s 存储抽象层次图" />

---

## 七、总结

| 概念 | 核心要点 | 适用场景 |
|---|---|---|
| `emptyDir` | Pod 内临时共享，Pod 销毁后消失 | sidecar 间数据共享、临时缓冲（KV Cache） |
| `hostPath` | 挂载节点文件系统，破坏可移植性 | 系统级 DaemonSet（日志收集、监控） |
| `gitRepo` | 启动时 clone 仓库，之后不同步 | 静态网站内容初始化 |
| 云存储 Volume | 直接引用底层存储，可移植性差 | 单集群快速验证，不推荐生产使用 |
| PersistentVolume | 集群级资源，管理员创建 | 预置存储池，供开发者按需申领 |
| PersistentVolumeClaim | Namespace 级资源，开发者创建 | 应用声明存储需求，与基础设施解耦 |
| StorageClass | 定义存储"配方"，触发动态供应 | 大规模集群自动化存储管理 |
| ReadWriteOnce | 单节点读写 | 数据库、单实例有状态服务 |
| ReadOnlyMany | 多节点并发只读 | **模型权重分发**、静态内容 |
| ReadWriteMany | 多节点并发读写 | 分布式训练 checkpoint（需 NFS/Ceph 支持） |

**核心设计原则回顾**：Kubernetes 存储抽象的演进路径是一条不断**解耦**的历程：从直接引用基础设施细节，到 PV/PVC 的角色分离，再到 StorageClass 的完全自动化。每一层抽象都是为了让应用开发者能够专注于"我需要什么"，而不是"基础设施如何提供"。

在 AI 工作负载中，这套抽象尤其重要——模型训练和推理的存储需求差异悬殊（训练需要高吞吐 RWX，推理需要高并发 ROX），StorageClass 的动态供应机制让集群能够优雅地应对这种多样性。
