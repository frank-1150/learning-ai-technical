# AI Technical Learning Notes

A bilingual (Chinese/English) knowledge base for AI and machine learning — written in Markdown, maintained with LLMs, viewable in Obsidian, and published as a static site via [VitePress](https://vitepress.dev/) + GitHub Pages.

**[Read it live](https://frank-1150.github.io/learning-ai-technical/)**

## Why This Exists

1. For learning, ask AI to add a note about any materials you are interested in, and you can view, edit, and organize them later in obsidian.
2. For sharing, you can share the website with others, and they can view the notes without installing any software.
3. Easy to maintain and index. You can search the notes by title, tag, or section. You can also use the tools/skills to maintain this repo.

The entire knowledge base is **LLM-maintained**: articles are drafted, organized, and enhanced by Claude, then reviewed and published by a human. Think of it as a minimal, open-source version of what Andrej Karpathy describes as an "LLM knowledge base."

## Features & Workflow

### Intelligent Automation

- [publish-article.md](./.claude/commands/publish-article.md): Ask Claude to write you an article based on the website or materials, you can put the materials under the `private/` folder so that they won't be shared to the public. After writing the article, you need to run this command to publish it.
- [organize-note.md](./.claude/commands/organize-note.md): Ask Claude to organize the notes, it will update the index.md and sidebar config. You can also use this command to update the nav dropdown.

### Static Website Hosting

VitePress hosts the static website based on your markdown files, and it is deployed on GitHub Pages.

## How to Use It

### 1. Clone and install

**Prerequisite:** Ensure you have Node.js installed (v18.0 or higher is recommended for VitePress).

```bash
git clone https://github.com/frank-1150/learning-ai-technical.git
cd learning-ai-technical
npm install
```

### 2. Run the dev server

```bash
npm run docs:dev       # Dev server with hot reload
npm run docs:build     # Production build
npm run docs:preview   # Preview production build
```

Open `http://localhost:5173/learning-ai-technical/` in your browser.

### 3. Write a new article

Use [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to draft articles from source materials:

1. Put your reference materials (PDFs, web clippings, notes) in the `private/` folder — this folder is gitignored and will never be published.
2. Ask Claude to write an article based on your materials, for you to view, learn or edit later in obsidian.
3. In your terminal running the Claude Code CLI, run `/publish-article <path-to-article>` to register it into the knowledge base (frontmatter, index, sidebar, nav, and build check — all handled automatically).

Or organize an existing draft note:

```bash
/organize-note <path-or-paste-content>
```

_(Make sure to run these slash commands directly inside your Claude Code CLI prompt)_

### 4. Edit with Obsidian

Open the `docs/` folder as an Obsidian vault. The vault is pre-configured to use standard markdown links, relative paths, and save images to `./images/`.

### 5. Deploy to your own GitHub Pages

1. Fork this repo.
2. Go to **Settings → Pages → Source** and select **"GitHub Actions"** (not "Deploy from a branch").
3. Update the `base` path in `docs/.vitepress/config/shared.mts` to match your repo name:
   ```ts
   base: '/your-repo-name/',
   ```
4. Push to `main` — the GitHub Actions workflow (`.github/workflows/deploy.yml`) will build and deploy automatically.

### 6. Star this repo!

If you find this useful, please give the repo a star — it helps others discover it too.

**[github.com/frank-1150/learning-ai-technical](https://github.com/frank-1150/learning-ai-technical)**

## Tech Stack

| Layer          | Tool                                 |
| -------------- | ------------------------------------ |
| Writing        | Obsidian + Claude                    |
| Framework      | VitePress                            |
| Hosting        | GitHub Pages                         |
| CI/CD          | GitHub Actions                       |
| Math           | MathJax (via markdown-it-mathjax3)   |
| Visualizations | Interactive HTML embedded via iframe |

## Project Structure

```text
docs/
  .vitepress/           # VitePress config and theme
  ai-applications/      # English: AI application topics
  machine-learning/     # English: ML topics
  zh/                   # Chinese translations (mirrored structure)
  public/               # Interactive HTML visualizations
```

## Contributing

Contributions are welcome! Whether it's fixing a typo, improving a translation, or adding a new article — feel free to open a PR.

## License

[MIT](./LICENSE)
