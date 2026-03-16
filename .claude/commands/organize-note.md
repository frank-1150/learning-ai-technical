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
   - YAML frontmatter: title, description, tags
   - Proper heading hierarchy (h1 matches title, h2+ for sections)
   - Standard markdown image links `![alt](./images/filename.png)` (NOT Obsidian wiki-links)
   - Code blocks with language identifiers
   - VitePress custom containers (`::: tip`, `::: warning`, `::: danger`) where appropriate

5. Place the file in the correct location under `docs/`.

6. If the user wants an English translation, also create the English version under `docs/en/` in the mirrored path.

7. Update the sidebar config in `docs/.vitepress/config/zh.mts` (and `en.mts` if English version was created) to include the new article.

Example file paths for a new RAG article:
- Chinese: `docs/ai-applications/rag/advanced-retrieval.md`
- English: `docs/en/ai-applications/rag/advanced-retrieval.md`

Always use relative image paths: `./images/filename.png`
Always include the frontmatter `tags` field for Obsidian tag search.
