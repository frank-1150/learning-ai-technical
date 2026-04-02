---
date: "2026-04-02"
title: "Ricursive Intelligence：AI for Chip Design & Chip Design for AI"
description: Lightspeed VC 访谈 Ricursive Intelligence 创始人笔记——两位 AlphaChip 发明人的故事、三阶段路线图，以及为什么芯片工程师是全世界最贵的建筑师
tags: [chip-design, ai-hardware, startup, lightspeed, ricursive]
---

# Ricursive Intelligence：AI for Chip Design & Chip Design for AI


**来源**：Lightspeed Venture Partners 系列访谈（YouTube）

AI is Designing the Chips that Power AI | Anna Goldie and Azalia Mirhoseini, Ricursive
https://youtu.be/VVwCyloDcL4?si=uTuBDb_6LoTc51rI
![](images/Pasted%20image%2020260402115216.png)

---

## 公司基本情况

**注意**：公司正式名称是 **Ricursive Intelligence**，不是 Recursive——这个拼写是刻意的，字母 i 强调"智能"（Intelligence）融入递归回路本身。

**创始人**：
- **Anna Goldie**（CEO）：AlphaChip 联合发明人，曾任 Google Brain 研究科学家、Anthropic 早期员工
- **Azalia Mirhoseini**（CTO）：AlphaChip 联合发明人，同样经历 Google Brain → Anthropic → Google 的完整路径

**AlphaChip 背景**：两人在 Google Brain 用深度强化学习攻克了芯片物理设计（floor planning）——这是芯片设计中最耗人力、最难自动化的一步。AlphaChip 已经商业落地，驱动了 Google TPU 的**连续四代设计**（包括运行 Gemini、Imagen 等模型的芯片），并被 MediaTek 等半导体公司对外采用。

---

## 融资情况

| 轮次 | 金额 | 领投 | 时间 |
|------|------|------|------|
| 种子轮 | $3500 万 | Sequoia Capital | 2025 年 12 月 |
| A 轮 | **$3 亿** | **Lightspeed Venture Partners** | 2026 年 1 月 |

A 轮估值 **$40 亿**，公司正式成立后**仅两个月**便完成。其他参与方：DST Global、NVentures（NVIDIA 旗下 VC）、Felicis Ventures、Radical AI。

**Lightspeed 的投资逻辑**（从他们的 blog post 里提炼）：① 创始团队有 production-scale 的真实验证；② 半导体设计市场的绝对规模；③ 从 Google DeepMind、NVIDIA、Apple、Cadence 聚拢来的顶级工程人才。

---

## 主持人的访谈技巧真的绝了

Host 发现两位创始人经常**互相帮对方补完一句没说完的话**（complete each other's sentences），两人配合得天衣无缝——然后 Host 用这个铺垫来发问：**"你们两个为什么会这么同步？"**

这个问题一下子引出了她们职业生涯里最精彩的一段故事，同时让参与访谈的两位很高兴，氛围也欢快起来：

- 同一天入职 Google Brain
- 同一天从 Google Brain 离职去 Anthropic
- 同一天从 Anthropic 离职，回到 Google，然后同一天再离职

这不是计划好的，是两个人对工作节奏和时机的判断真的高度一致。访谈用这种方式展示了她们作为联合创始人的默契，远比直接问"你们是怎么认识的"要生动、有趣得多。

Anna 作为 CEO 在整个访谈过程中盯着 host 讲话的那种专注感，60 秒内眼睛都不眨一下，让我想起了高中时期上课时大家都样子。

---

## 核心洞见

**AI 的本质方向：蒸馏、加速、自动化最有价值的工作**

最有价值的工作不一定是最多人做的工作——造芯片的工程师是全世界最贵的建筑师，他们的时间被花在了不需要人类智能才能完成的事情上。这就是 Ricursive 存在的逻辑。

**Vibe coding 扩大了整个市场**

之前只有约 2500 万人能写软件，有了 AI coding 工具之后，至少有 1 亿人能用工具写软件——coding 工具的天花板被直接抬高了。Anna 和 Azalia 用这个类比来阐述 Ricursive 的市场逻辑：不是在争现有市场的份额，而是在创造一个新的更大的市场。

**Google 内部 repo 是全员可见的**

两位创始人在 Google 任职期间可以访问内部几乎所有代码库——这是 Google 内部的工程文化，对内部员工完全开放。对于芯片设计 AI 的训练数据质量而言，这是极其稀缺的 unfair advantage。OMG。

**Amazing validation**

AlphaChip 不是 demo 或 paper，它已经参与 Google TPU 的最核心的生产环境里四代了。

---

## 三阶段路线图

### Phase 1：加速现有的芯片设计流程

把人类工程师几个月才能完成的 floor planning、physical design 等任务，压缩到小时级别。核心客户是已有芯片设计能力的公司，帮他们把速度拉快、成本拉低。

### Phase 2：芯片设计平台——让更多公司拥有自己的芯片

今天大量 AI 应用公司跑在通用 CPU/GPU 上，因为定制芯片的门槛太高——你需要一支世界上最贵的工程师团队。

Ricursive 的 Phase 2 是做 **design-less**（类比台积电的 fabless 模式）：任何在一定规模上运行自己软件的公司，都应该拥有为自己量身定制的芯片。客户只需要专注应用层，底层的硅设计交给 Ricursive。

> "Any company that runs their software at scale properly deserves their own customized silicon."

### Phase 3：AI 设计 AI 的芯片，递归自我提升

用强化学习训练出更好的芯片设计方法 → 用这个方法造更强的芯片 → 用更强的芯片跑更好的 AI → 继续迭代。这是名字里"Ricursive"真正指向的东西。

这有点像 Jensen Huang 说的 **extreme co-design 的平民版**：软件和硬件不应该分开优化，绝大多数今天的软件——包括一些世界上最优秀的 AI 模型——都被困在通用硬件上，无法发挥它们本来应有的性能上限。

---

## 结构性颠覆（Constructive Disruption）

Host 说：他做研究的时候发现，Ricursive 的市场不只是"半导体设计工具"市场（传统 EDA 软件，Cadence、Synopsys 那个市场）。

真正的市场规模分布是：**2/3 的钱花在工程师团队，1/3 花在工具上**。Ricursive 同时吃这两块——不仅替代工具，还替代人力。更难的问题甚至不是钱，而是**优秀的芯片工程师根本不够用**，全球都在抢同一批人。这个跟我们前面的核心观点第一点很相呼应，不仅替代了人力，而更加是补充了某些公司某些市场招不到的工程师经验。蒸馏浓缩了芯片设计工程师处理某些子类问题、垂类问题的方法、经验，然后规模化地应用。

这才是真正的结构性机会：不仅是提供一个更好的工具，更加是扩展了用户群体，把之前做不了芯片的现在也能做芯片了，招不到工程师的现在可以用他们的解决方案了，扩大了市场。
