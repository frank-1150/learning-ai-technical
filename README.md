# AI Technical Learning Notes

A bilingual (Chinese/English) knowledge base for AI and machine learning topics, built with [VitePress](https://vitepress.dev/) and compatible with [Obsidian](https://obsidian.md/).

## Topics

- **AI Applications** — RAG, Agents, Prompt Engineering
- **Machine Learning** — PyTorch, Tensors, Neural Networks

## Local Development

```bash
# Install dependencies
npm install

# Start dev server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Editing with Obsidian

Open the `docs/` folder as an Obsidian vault. The vault is pre-configured to:
- Use standard markdown links (not wiki-links)
- Use relative paths for links
- Save image attachments to `./images/` subfolder

## Project Structure

```
docs/                      # VitePress source + Obsidian vault
  .vitepress/              # VitePress config and theme
  ai-applications/         # Chinese: AI application topics
  machine-learning/        # Chinese: ML topics
  en/                      # English translations
```

## Deployment

Automatically deploys to GitHub Pages on push to `main` via GitHub Actions.
