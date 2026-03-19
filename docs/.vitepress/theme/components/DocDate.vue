<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
import { computed } from 'vue'

const { Layout } = DefaultTheme
const { frontmatter, lang } = useData()

const isZh = computed(() => lang.value.startsWith('zh'))

function formatDate(dateStr) {
  if (!dateStr) return ''
  // YAML parses bare dates (2026-01-01) as Date objects at midnight UTC,
  // not strings — extract UTC parts to avoid String(Date) mangling.
  let year, month, day
  if (dateStr instanceof Date) {
    year = dateStr.getUTCFullYear()
    month = dateStr.getUTCMonth() + 1
    day = dateStr.getUTCDate()
  } else {
    ;[year, month, day] = String(dateStr).split('-').map(Number)
  }
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
