import { defineConfig } from 'vitepress'

export const shared = defineConfig({
  title: 'AI Learning Notes',

  // Set to repo name for GitHub Pages. Change to '/' if using custom domain.
  base: '/learning-ai-technical/',

  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['meta', { name: 'theme-color', content: '#5b8dff' }],
    ['meta', { name: 'og:type', content: 'website' }],
  ],

  markdown: {
    lineNumbers: true,
    image: {
      lazyLoading: true
    },
    math: true
  },

  themeConfig: {
    socialLinks: [
      { icon: 'github', link: 'https://github.com/frank-1150/learning-ai-technical' }
    ],
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: { selectText: '选择', navigateText: '切换' }
              }
            }
          }
        }
      }
    }
  }
})
