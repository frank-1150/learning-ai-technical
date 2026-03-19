<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'

const { Layout } = DefaultTheme
const { frontmatter, lang } = useData()

function formatDate(dateStr) {
  if (!dateStr) return ''
  const [year, month, day] = String(dateStr).split('-').map(Number)
  const d = new Date(year, month - 1, day)
  const isZh = lang.value.startsWith('zh')
  return d.toLocaleDateString(isZh ? 'zh-CN' : 'en-US', {
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
  </Layout>
</template>
