---
description: Run the post-article checklist for a newly written VitePress article — verify frontmatter, update section index.md, update sidebar config, build to catch dead links, then commit.
---

The user has finished writing an article and wants to register it properly into the knowledge base. The article path is: $ARGUMENTS

Follow these steps in order:

## Step 1 — Read the article

Read the article file. Confirm it exists and note:
- Its path (e.g. `docs/ai-applications/how-i-use-ai/my-article.md`)
- Its section (the parent folder, e.g. `how-i-use-ai`)
- Whether an English counterpart exists under `docs/en/`

## Step 2 — Verify frontmatter

The article must have all of:
```yaml
---
date: YYYY-MM-DD        # today if missing
title: "..."
description: "..."
tags: [tag1, tag2]
---
```
If any field is missing, add it. Use today's date (from `currentDate` context) if `date` is absent.

## Step 3 — Update the section index.md

The section landing page (e.g. `docs/ai-applications/how-i-use-ai/index.md`) must list the article under its `## 文章` (or `## Articles`) heading.

- Read the section index.
- If the article is not already linked, add a bullet in this format:
  `- [Article title](./filename.md) — one-line description`
- Use URL-encoded spaces (`%20`) for filenames with spaces.
- If there's an English section index, update it too (or add `*To be added*` placeholder if no English article exists).

## Step 4 — Update sidebar config

Open `docs/.vitepress/config/zh.mts` and find the sidebar block for the article's section.

- If the article is already listed, skip.
- If not, add an entry:
  `{ text: 'Article title', link: '/section/subsection/filename' }`
  **Do not** URL-encode the link — VitePress handles filenames with spaces directly.

If an English article exists, do the same in `docs/.vitepress/config/en.mts`.

## Step 5 — Build

Run `npm run docs:build`. VitePress treats dead links as build errors.

- If the build passes, proceed.
- If it fails with a dead link error, fix the broken link before continuing.

## Step 6 — Commit

Stage only the changed files (the article itself if frontmatter was added, section index.md, sidebar config). Commit with a message like:

```
fix: register <article-title> into index and sidebar
```

Report what was changed and confirm the build passed.
