<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{
  src: string
  height?: string
  title?: string
}>()

const iframeSrc = computed(() => withBase(props.src))
const iframeHeight = computed(() => props.height || '400px')
</script>

<template>
  <div class="html-visualization">
    <div v-if="title" class="viz-title">{{ title }}</div>
    <iframe
      :src="iframeSrc"
      :style="{ height: iframeHeight }"
      frameborder="0"
      loading="lazy"
      sandbox="allow-scripts allow-same-origin"
    />
  </div>
</template>

<style scoped>
.html-visualization {
  margin: 16px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.html-visualization iframe {
  width: 100%;
  display: block;
}
.viz-title {
  padding: 8px 16px;
  font-weight: 600;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
}
</style>
