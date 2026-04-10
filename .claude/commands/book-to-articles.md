---
description: "将一本 PDF 技术书籍拆分为主题栏目，并为中文 VitePress 站点生成完整的阅读笔记，包含深度技术文章、PDF 图片截图和交互式可视化。用法：/book-to-articles <PDF路径> <栏目slug>"
---

将一本技术书籍转化为结构化的阅读笔记栏目。参数：$ARGUMENTS

第一个参数是 PDF 路径（相对于项目根目录），第二个参数是栏目的 slug（如 `kubernetes-in-action`）。

示例：`/book-to-articles private/books/MyBook.pdf my-book-slug`

---

## Phase 1：读取目录并规划拆分

首先用 Python 读取 PDF 的书签/目录：

```python
import pymupdf, os

pdf_path = "<第一个参数>"
doc = pymupdf.open(pdf_path)
print(f"Total pages: {doc.page_count}")
toc = doc.get_toc()
for level, title, page in toc:
    print(f"{'  '*(level-1)}[L{level}] p{page} {title}")
```

读取输出后，**按主题而非按 Part/Chapter 机械拆分**，将内容聚合成 6-10 篇主题文章，每篇覆盖 1-4 个相关章节。目标是：

- 每篇文章有清晰的主题焦点
- 相关联的章节合并在一起（不要把同一个主题切断）
- 每篇对应的原书页数在 30-120 页之间

拆分后创建各段 PDF 文件：
- `00-Front-Matter.pdf` — 封面到 Preface（用于写书籍总览页）
- `01-<theme-name>.pdf`、`02-<theme-name>.pdf` ... 每个主题一个文件
- `99-Back-Matter.pdf` — Appendices、Index（如果存在）

保存到 `private/books/<slug>-parts/`。

```python
import pymupdf, os

pdf_path = "<第一个参数>"
out_dir = "private/books/<slug>-parts"
os.makedirs(out_dir, exist_ok=True)

doc = pymupdf.open(pdf_path)

# 按规划的页码范围拆分（PDF 页码 0-indexed，toc 里是 1-indexed）
segments = [
    ("00-Front-Matter", 1, <preface_end>),
    ("01-<theme>", <start>, <end>),
    # ...
    ("99-Back-Matter", <appendix_start>, doc.page_count),
]

for name, start_p, end_p in segments:
    out = pymupdf.open()
    out.insert_pdf(doc, from_page=start_p-1, to_page=end_p-1)
    out.save(f"{out_dir}/{name}.pdf")
    print(f"Saved {name}.pdf ({end_p - start_p + 1} pages)")
```

---

## Phase 2：确定站点结构

检查现有站点结构，决定新栏目的位置：

1. 读取 `docs/.vitepress/config/zh.mts`，了解现有导航和侧边栏
2. 新栏目路径：`docs/zh/book-notes/<slug>/`
3. 图片路径：`docs/public/book-notes/<slug>/images/`
4. 可视化文件路径：`docs/public/book-notes/<slug>/visualizations/`
5. 创建这三个目录

---

## Phase 3：并行派发子 Agent 写文章

**在单条消息中同时启动所有 Agent（后台模式）**，每个 Agent 负责一个主题文件。

每个 Agent 的任务：
1. 用 Read 工具读取对应的主题 PDF（指定 pages 参数分批读完全部内容）
2. 提取关键 PDF 图片（见下方截图指南）
3. 撰写一篇中文技术文章（2500-4000 字）
4. 创建 1-2 个交互式 HTML 可视化

### Agent 提示模板

```
你是为一个 VitePress 技术博客写中文阅读笔记的 agent。

## 你的任务
阅读这本书的主题文件（路径：private/books/<slug>-parts/<N>-<name>.pdf），
然后写一篇深度技术文章，配 PDF 截图和 1-2 个交互式可视化。

## 文章写到
docs/zh/book-notes/<slug>/<article-slug>.md

frontmatter 格式：
---
date: <today>
title: "<中文标题>"
description: "<一句话描述>"
tags: [<相关标签>]
---

文章结构要求：
1. **开篇**：先写一个「本文覆盖章节」的引导块，例如：
   > 本文对应原书 **第 X、Y、Z 章**，覆盖：[章节名1]、[章节名2]、[章节名3]

2. **正文**：对每个核心概念，必须同时讲清楚：
   - **是什么**（What）：准确定义，附代码/YAML 示例
   - **为什么**（Why）：这个设计解决了什么问题，不用它会怎样
   - 在适当地方用 AI 推理基础设施场景举例（如：模型服务、GPU 节点调度、权重存储等），但不要暴露文章是为某个特定读者写的——写成对所有读者都有帮助的通用深度文章

3. **结尾**：一个小结表格，列出本文覆盖的所有概念及其核心作用

其他格式要求：
- 中文，技术但易懂
- 用表格、代码块、callout boxes（> [!tip]、> [!warning]、> [!note]）
- 2500-4000 字实质内容
- 在合适位置插入从 PDF 提取的图片（见下方）

## 提取 PDF 中的关键图片

用以下 Python 代码提取书中的重要图表，保存为 PNG：

```python
import pymupdf, os

doc = pymupdf.open("private/books/<slug>-parts/<N>-<name>.pdf")
out_dir = "docs/public/book-notes/<slug>/images"
os.makedirs(out_dir, exist_ok=True)

# 方法一：整页渲染（适合包含完整图表的页面）
page = doc[<page_index>]  # 0-indexed
mat = pymupdf.Matrix(2, 2)  # 2x 分辨率，清晰
pix = page.get_pixmap(matrix=mat)
pix.save(f"{out_dir}/<描述性名称>.png")

# 方法二：裁剪局部区域（适合只取页面某一部分）
rect = pymupdf.Rect(x0, y0, x1, y1)  # 坐标从左上角开始，单位 pt
clip = page.get_pixmap(matrix=mat, clip=rect)
clip.save(f"{out_dir}/<描述性名称>.png")
```

提取原则：
- 优先提取架构图、流程图、对比图，跳过纯文字页
- 文件名用英文描述性命名，如 `pod-lifecycle.png`、`service-routing.png`
- 每篇文章提取 2-5 张有价值的图

在文章中用标准 Markdown 图片语法引用：
```markdown
![图片说明](/book-notes/<slug>/images/<filename>.png)
```

## 可视化写到
docs/public/book-notes/<slug>/visualizations/<name>.html

可视化规则（必须遵守）：
- 深色主题默认，body 加 .light 类时切换浅色
- CSS 变量：
  :root { --bg:#1e1e2e; --fg:#cdd6f4; --muted:#6c7086; --surface:#181825;
          --border:#45475a; --overlay:#313244; --green:#a6e3a1; --red:#f38ba8;
          --blue:#89b4fa; --yellow:#f9e2af; --mauve:#cba6f7; --peach:#fab387 }
  .light { --bg:#eff1f5; --fg:#4c4f69; --muted:#8c8fa1; --surface:#e6e9ef;
           --border:#ccd0da; --overlay:#dce0e8; --green:#40a02b; --red:#d20f39;
           --blue:#1e66f5; --yellow:#df8e1d; --mauve:#8839ef; --peach:#fe640b }
- 字体：'SF Mono', 'Fira Code', 'Cascadia Code', monospace
- 脚本顺序（严格遵守，否则 TDZ 报错）：
  1. 所有 const/let/function 声明
  2. 最后调用 syncTheme()，然后注册 MutationObserver

syncTheme 模板：
function syncTheme() {
  try {
    const html = window.parent.document.documentElement;
    document.body.classList.toggle('light', !html.classList.contains('dark'));
  } catch(e) {}
  render(); // 或 rebuild()
}
syncTheme();
try { new MutationObserver(syncTheme).observe(
  window.parent.document.documentElement,
  { attributes: true, attributeFilter: ['class'] }
); } catch(e) {}

在文章中用以下方式引用可视化：
<HtmlVisualization src="/book-notes/<slug>/visualizations/<name>.html" height="NNNpx" title="标题" />

现在开始：先读 PDF，再提取图片，再写文章，最后写可视化。全部写完后报告完成。
```

### 同时还需要一个 Agent 写 index 页面

该 Agent 负责：
1. 读取 `00-Front-Matter.pdf`（前言、目录等）
2. 写 `docs/zh/book-notes/index.md`（如不存在）— 技术书籍阅读栏目首页
3. 写 `docs/zh/book-notes/<slug>/index.md` — 书籍总览页，包含：
   - 作者和书籍背景介绍
   - 全书主题文章结构总览表格（链接到各篇文章）
   - 「为什么读这本书」一节：简述这本书在技术栈中的地位
4. 创建一个书籍结构可视化 `book-structure.html`（可折叠的主题/章节导航图）

---

## Phase 4：等待 Agent 完成后收尾

所有 Agent 完成后：

### 4.1 检查文件
```bash
ls docs/zh/book-notes/<slug>/
ls docs/public/book-notes/<slug>/images/
ls docs/public/book-notes/<slug>/visualizations/
```
确认所有文章、图片和可视化都已创建。若有缺失，手动补写。

### 4.2 更新 VitePress 配置

在 `docs/.vitepress/config/zh.mts` 中：

**导航栏**（`nav()` 函数）：
若 `技术书籍阅读` 还不存在，在最后加：
```typescript
{
  text: '技术书籍阅读',
  items: [
    { text: '<书名>', link: '/zh/book-notes/<slug>/' }
  ]
}
```
若已存在，在 items 中追加新书。

**侧边栏**（`sidebar()` 函数）：
若 `/zh/book-notes/` 路由还不存在，添加：
```typescript
'/zh/book-notes/': [
  {
    text: '技术书籍阅读',
    items: [
      { text: '概览', link: '/zh/book-notes/' },
      {
        text: '<书名>',
        collapsed: false,
        items: [
          { text: '书籍总览', link: '/zh/book-notes/<slug>/' },
          { text: '主题一 — <名称>', link: '/zh/book-notes/<slug>/<article1-slug>' },
          // ... 每篇文章一条
        ]
      }
    ]
  }
]
```
若路由已存在，只在书籍列表中追加新书。

### 4.3 构建验证
```bash
npm run docs:build
```
构建必须通过（无死链、无构建错误）。若失败，修复后重新构建。

### 4.4 Commit
用 `/commit` 命令提交所有新文件和配置变更。

---

## 可视化选题指南

根据章节内容选择合适的可视化形式：

| 内容类型 | 推荐可视化 |
|---------|-----------|
| 概念比较（A vs B） | 交互式切换卡片，点击在两种模式间切换 |
| 流程/管道 | 动画数据流图，带动画粒子或状态变化 |
| 层级/树形结构 | 可折叠的树形图 |
| 频谱/连续体 | 可拖动滑块，显示不同点位的属性 |
| 状态机 | 带按钮触发的状态转移动画 |
| 时间线 | 带步骤控制的时序图 |
| 画廊/卡片集 | 可点击展开的卡片网格 |

---

## 注意事项

- **PDF 路径**：用 `pymupdf.open()` 时传绝对路径
- **并行 Agent**：必须在一条消息里同时启动，否则会串行执行
- **Agent 权限**：Agent 有时会遇到 Write 权限问题，若某个 Agent 报告无法写入，手动从其输出中提取内容并写入
- **内容质量**：Agent 若没有读到实际 PDF 内容（只根据 TOC 推断），文章会比较空洞。确保 Agent 真正读取了 PDF 并引用了书中原文
- **图片质量**：用 2x Matrix 渲染以确保清晰度；若图表页面有大量留白，用 Rect 裁剪
- **可视化调试**：CLAUDE.md 中列出了常见可视化陷阱（TDZ、lang 属性、日期序列化等），若可视化报错请对照检查
