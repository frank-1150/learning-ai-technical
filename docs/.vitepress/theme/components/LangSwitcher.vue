<script setup>
import { computed } from 'vue'
import { useData, useRouter, withBase } from 'vitepress'

const { lang, page } = useData()
const router = useRouter()

const isZh = computed(() => lang.value.startsWith('zh'))

function switchTo(targetLang) {
  const rel = page.value.relativePath // e.g. 'ai-applications/index.md' or 'zh/ai-applications/index.md'
  let targetRel
  if (targetLang === 'zh') {
    targetRel = rel.startsWith('zh/') ? rel : 'zh/' + rel
  } else {
    targetRel = rel.startsWith('zh/') ? rel.slice(3) : rel
  }
  // Convert file path to clean URL: strip .md, strip trailing /index
  const url = withBase('/' + targetRel.replace(/\.md$/, '').replace(/\/index$/, '/').replace(/^index$/, ''))
  router.go(url)
}
</script>

<template>
  <div class="lang-switcher">
    <button
      :class="{ active: !isZh }"
      @click="switchTo('en')"
    >EN</button>
    <button
      :class="{ active: isZh }"
      @click="switchTo('zh')"
    >中文</button>
  </div>
</template>

<style scoped>
.lang-switcher {
  display: flex;
  align-items: center;
  gap: 2px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 20px;
  padding: 3px;
  margin-right: 8px;
}

.lang-switcher button {
  padding: 3px 12px;
  border-radius: 16px;
  border: none;
  background: transparent;
  color: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  line-height: 1.5;
}

.lang-switcher button:hover {
  color: var(--vp-c-text-1);
}

.lang-switcher button.active {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
}
</style>
