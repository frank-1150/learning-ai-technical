---
description: 读取 private/zsxq/downloads/ 下的知识星球文件，提炼内容并整合到 industry-views 行业观点板块。用法：/zsxq-to-industry-views YYYY-MM
---

用户想把知识星球下载的文件整理到行业观点知识库。目标月份：$ARGUMENTS

按以下步骤执行：

## Step 1 — 扫描下载目录

用 Glob 或 Bash 列出 `private/zsxq/downloads/$ARGUMENTS/` 下的所有文件。

如果该目录不存在或为空，告知用户先运行 `cd private/zsxq && python3 downloader.py` 下载文件，然后停止。

## Step 2 — 读取并分析文件内容

逐一读取所有文件（Read tool 支持 PDF）。对每个文件：
- 识别文件主题和核心内容
- 提取以下三类信息：
  1. **行业共识**：被多方认可的趋势/判断（不是某一个人的观点，而是行业普遍认知）
  2. **人士观点**：具体人物/机构的观点、预测、发言（需标注来源）
  3. **事实数据**：具体事件、数据、产品发布、政策等可核实的事实

## Step 3 — 检查目标文件是否已存在

检查以下三个文件是否已存在：
- `docs/ai-applications/industry-views/consensus/$ARGUMENTS.md`
- `docs/ai-applications/industry-views/perspectives/$ARGUMENTS.md`
- `docs/ai-applications/industry-views/facts/$ARGUMENTS.md`

**如果已存在**：读取现有内容，将新提炼的内容追加进去（避免重复）。

**如果不存在**：新建文件，使用以下 frontmatter 格式：

```yaml
---
date: YYYY-MM-01
title: 行业共识 · YYYY-MM      # perspectives 改为"行业人士观点"，facts 改为"事实收集"
description: （一句话描述本月主要内容）
tags: [industry-views, consensus]   # consensus/perspectives/facts 对应不同 tag
---
```

## Step 4 — 写入内容

按各文件的格式写入提炼的内容：

### consensus/YYYY-MM.md 格式

```markdown
## 1. 共识标题（加粗）

- **形成时间**：大约 YYYY 年 Q? 季度
- **来源**：文件名或作者（日期）

具体描述，说明这个共识的内容、背景和意义。
```

### perspectives/YYYY-MM.md 格式

```markdown
## 姓名 · 职位/机构

- **来源**：文件名（YYYY-MM-DD）

**观点主题**：具体观点内容，尽量保留原话的核心意思。
```

### facts/YYYY-MM.md 格式

```markdown
## 分类（如：模型与产品 / 算力与基础设施 / 行业动态）

- **YYYY-MM-DD**：事实描述（来源：文件名）
- **YYYY-MM-DD**：事实描述（来源：文件名）
```

## Step 5 — 更新 index.md 归档列表

如果是新建的月度文件，在对应的 index.md 归档列表中加入新月份链接：

- `docs/ai-applications/industry-views/consensus/index.md`
- `docs/ai-applications/industry-views/perspectives/index.md`
- `docs/ai-applications/industry-views/facts/index.md`

在 `## 归档` 下追加：
```markdown
- [YYYY-MM](./YYYY-MM)
```

注意：不需要更新 VitePress sidebar 配置，也不需要同步 `docs/en/`（原始内容为中文）。

## Step 6 — 汇报结果

告诉用户：
- 读取了哪些文件
- 各类别各提炼了多少条内容
- 写入了哪些文件（新建/追加）
- 有哪些内容因为信息不足被跳过
