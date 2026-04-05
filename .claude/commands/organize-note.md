---
description: Organize a draft note into the knowledge base structure
---

The user has a draft note or raw content that needs to be organized into the VitePress knowledge base at $ARGUMENTS.

Your task:

1. Read the draft content the user provides (either pasted text or a file path).

2. Determine the appropriate category:
   - `ai-applications` — for RAG, agents, prompt engineering, and other AI application topics
   - `machine-learning` — for PyTorch, tensors, neural networks, and other ML topics
   - Or suggest creating a new top-level category if the content doesn't fit.

3. Determine the appropriate topic subfolder (rag, agents, prompt-engineering, pytorch, tensors, neural-networks), or suggest creating a new one.

4. Format the content as a proper VitePress-compatible markdown file with:
   - YAML frontmatter: title, description, tags, date (quoted string `"YYYY-MM-DD"` — bare dates cause VitePress serialization bugs)
   - Proper heading hierarchy (h1 matches title, h2+ for sections)
   - Standard markdown image links `![alt](./images/filename.png)` (NOT Obsidian wiki-links)
   - Code blocks with language identifiers
   - VitePress custom containers (`::: tip`, `::: warning`, `::: danger`) where appropriate

5. Place the file in the correct location:
   - Chinese articles go under `docs/zh/` (e.g. `docs/zh/ai-applications/rag/advanced-retrieval.md`)
   - English articles go under `docs/` (e.g. `docs/ai-applications/rag/advanced-retrieval.md`)

6. If the user wants a translation, also create the counterpart version in the other locale with a mirrored path.

7. Update the section index.md to list the new article:
   - Read the section landing page (e.g. `docs/zh/ai-applications/rag/index.md`)
   - Add a bullet: `- [Article title](./filename.md) — one-line description`
   - If the counterpart locale's section index exists, update it too (or add `*To be added*` placeholder).

8. Update the sidebar config in `docs/.vitepress/config/zh.mts` (paths start with `/zh/`) and/or `docs/.vitepress/config/en.mts` (paths start with `/`) to include the new article.

9. Check the nav dropdown (the `nav()` function in `zh.mts` and `en.mts`). If the article's section is not yet listed in the nav `items` arrays, add it. Both locale configs must stay in sync — every section in one must appear in the other.

10. Run `npx vitepress build docs` to verify no dead links. Fix any errors before finishing.

Always use relative image paths: `./images/filename.png`
Always include the frontmatter `tags` field for Obsidian tag search.
