# Project Notes

## VitePress Bilingual Structure

This project has a **bilingual (zh/en)** VitePress site. When adding or removing pages:

- **Always update both locales**: Chinese pages go in `docs/`, English pages go in `docs/en/` with mirrored paths.
- **Update all config files**: `docs/.vitepress/config/zh.mts` (nav + sidebar) and `docs/.vitepress/config/en.mts` (nav + sidebar).
- **Update both index.md files**: `docs/machine-learning/index.md` and `docs/en/machine-learning/index.md` (or whichever section).
- **VitePress treats dead links as build errors** — if you delete a page, you must remove all links to it across both locales, or the CI build will fail.
- Interactive HTML visualizations go in `docs/public/` and are referenced via the `<HtmlVisualization>` Vue component.

## GitHub Actions Deployment

- The deploy workflow is at `.github/workflows/deploy.yml`.
- GitHub Pages source must be set to **"GitHub Actions"** (not "Deploy from a branch") to avoid a second `pages-build-deployment` workflow overwriting the VitePress build with plain Jekyll HTML.
- The Node.js 20 deprecation warning for actions is a **warning only** — do NOT set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` as it breaks the actions that aren't Node 24 compatible yet.
