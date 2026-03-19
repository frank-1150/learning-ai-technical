# Project Notes

## VitePress Bilingual Structure

This project has a **bilingual (zh/en)** VitePress site. When adding or removing pages:

- **Always update both locales**: Chinese pages go in `docs/`, English pages go in `docs/en/` with mirrored paths.
- **Update all config files**: `docs/.vitepress/config/zh.mts` (nav + sidebar) and `docs/.vitepress/config/en.mts` (nav + sidebar).
- **Update both index.md files**: `docs/machine-learning/index.md` and `docs/en/machine-learning/index.md` (or whichever section).
- **VitePress treats dead links as build errors** — if you delete a page, you must remove all links to it across both locales, or the CI build will fail.
- Interactive HTML visualizations go in `docs/public/` and are referenced via the `<HtmlVisualization>` Vue component.

## Interactive Visualization Gotchas

- Visualizations are loaded in `<iframe sandbox="allow-scripts allow-same-origin">`. Accessing `window.parent.document` may throw due to sandbox restrictions — always wrap in `try/catch`, but **never put render logic inside the try block** or it will be silently skipped on error.
- **YAML parses bare dates as `Date` objects**: `frontmatter.date` for a field like `date: 2026-01-01` (unquoted) is a `Date` object, not a string. `String(Date)` produces `"Thu Jan 01 2026..."` which cannot be split on `-` for year/month/day. Always check `dateStr instanceof Date` and use `getUTCFullYear/Month/Date` to extract parts. Using UTC getters avoids a timezone-shift issue (YAML dates are midnight UTC).
- **Lang attribute is BCP 47, not a short code**: `window.parent.document.documentElement.lang` returns `'zh-CN'` or `'en-US'`, not `'zh'`/`'en'`. If you use it as a key into a translation object `T`, always normalize first: `lang.startsWith('zh') ? 'zh' : 'en'`. Using the raw value silently produces `T['zh-CN'] === undefined`, which crashes any property access on it with a TypeError.
- **Temporal Dead Zone (TDZ)**: `const`/`let` variables are NOT hoisted. If a function like `syncTheme()` calls `render()`/`rebuild()` which reference `const`/`let` variables, `syncTheme()` must be called **after** all those declarations. Putting it at the top of `<script>` will cause a silent `ReferenceError: Cannot access 'X' before initialization`.
- Correct pattern for visualization scripts:
  ```
  <script>
  // 1. All const/let declarations and function definitions first
  const data = [...];
  function render() { /* uses data */ }

  // 2. Theme sync and initial render LAST
  function syncTheme() {
    try { /* read parent theme */ } catch(e) {}
    render();
  }
  syncTheme();
  try { new MutationObserver(syncTheme).observe(...); } catch(e) {}
  </script>
  ```

## GitHub Actions Deployment

- The deploy workflow is at `.github/workflows/deploy.yml`.
- GitHub Pages source must be set to **"GitHub Actions"** (not "Deploy from a branch") to avoid a second `pages-build-deployment` workflow overwriting the VitePress build with plain Jekyll HTML.
- The Node.js 20 deprecation warning for actions is a **warning only** — do NOT set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` as it breaks the actions that aren't Node 24 compatible yet.
