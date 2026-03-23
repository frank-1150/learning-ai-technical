<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
import { computed } from 'vue'
import LangSwitcher from './LangSwitcher.vue'

const { Layout } = DefaultTheme
const { frontmatter, lang } = useData()

const isZh = computed(() => lang.value.startsWith('zh'))

function formatDate(dateStr) {
  if (!dateStr) return ''
  // YAML bare dates are serialized by VitePress via JSON as ISO strings
  // ("2026-03-19T00:00:00.000Z"). Slicing to 10 chars gives "YYYY-MM-DD"
  // regardless of whether the original value was a string or a Date object.
  const [year, month, day] = String(dateStr).slice(0, 10).split('-').map(Number)
  const d = new Date(year, month - 1, day)
  return d.toLocaleDateString(isZh.value ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}
</script>

<template>
  <Layout>
    <template #nav-bar-content-after>
      <LangSwitcher />
    </template>
    <template #doc-before>
      <span v-if="frontmatter.date" class="article-date">
        {{ formatDate(frontmatter.date) }}
      </span>
    </template>
    <template #doc-after>
      <div class="star-cta">
        <a href="https://github.com/frank-1150/learning-ai-technical" target="_blank" rel="noopener">
          ⭐ {{ isZh ? '如果这篇文章有帮助，欢迎给这个项目点个 Star' : 'If this was helpful, consider starring the repo on GitHub' }}
        </a>
      </div>
    </template>
  </Layout>
</template>
