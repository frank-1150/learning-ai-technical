---
description: Run the post-article checklist for a newly written VitePress article — verify frontmatter, update section index.md, update sidebar config, build to catch dead links, then commit.
---

The user has finished writing an article and wants to register it properly into the knowledge base. The article path is: $ARGUMENTS

Follow these steps in order:

## Step 1 — Read the article

Read the article file. Confirm it exists and note:
- Its path (e.g. `docs/zh/ai-applications/how-i-use-ai/my-article.md`)
- Its section (the parent folder, e.g. `how-i-use-ai`)
- Its locale: files under `docs/zh/` are Chinese, files directly under `docs/` (not `docs/zh/`) are English
- Whether a counterpart exists in the other locale (`docs/` ↔ `docs/zh/`, mirrored path)

## Step 2 — Verify frontmatter

The article must have all of:
```yaml
---
date: "YYYY-MM-DD"       # quoted string — bare dates cause VitePress serialization bugs (see CLAUDE.md)
title: "..."
description: "..."
tags: [tag1, tag2]
---
```
If any field is missing, add it. Use today's date (from `currentDate` context) if `date` is absent. Always quote the date value.

## Step 3 — Update the section index.md

The section landing page (e.g. `docs/zh/ai-applications/how-i-use-ai/index.md`) must list the article under its `## 文章` (or `## Articles`) heading.

- Read the section index.
- If the article is not already linked, add a bullet in this format:
  `- [Article title](./filename.md) — one-line description`
- Use URL-encoded spaces (`%20`) for filenames with spaces.
- If the counterpart locale's section index exists, update it too (or add `*To be added*` placeholder if no translated article exists).

## Step 4 — Update sidebar config

Sidebar config files:
- Chinese sidebar: `docs/.vitepress/config/zh.mts` (paths start with `/zh/`)
- English sidebar: `docs/.vitepress/config/en.mts` (paths start with `/`)

Find the sidebar block for the article's section.

- If the article is already listed, skip.
- If not, add an entry:
  `{ text: 'Article title', link: '/zh/section/subsection/filename' }`
  **Do not** URL-encode the link — VitePress handles filenames with spaces directly.

If a counterpart article exists in the other locale, update that config too.

## Step 5 — Check nav dropdown

The nav dropdown (the `nav()` function in `zh.mts` and `en.mts`) lists sections under "AI Applications" / "AI 应用" and "Machine Learning" / "机器学习". If the article belongs to a section that is **not yet in the nav dropdown**, add it.

- Compare the sidebar sections with the nav `items` arrays in both `zh.mts` and `en.mts`.
- Both locale configs must stay in sync — every section in one must appear in the other.

## Step 6 — Build

Run `npx vitepress build docs` (or `/opt/homebrew/bin/npx vitepress build docs` if npx is not on PATH). VitePress treats dead links as build errors.

- If the build passes, proceed.
- If it fails with a dead link error, fix the broken link before continuing.

## Step 7 — Commit

Stage only the changed files (the article itself if frontmatter was added, section index.md, sidebar config). Commit with a message like:

```
fix: register <article-title> into index and sidebar
```

Report what was changed and confirm the build passed.
