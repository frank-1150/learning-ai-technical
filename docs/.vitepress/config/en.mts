import { defineConfig, type DefaultTheme } from 'vitepress'

export const en = defineConfig({
  lang: 'en-US',
  description: 'AI Technical Learning Notes',

  themeConfig: {
    nav: nav(),
    sidebar: sidebar(),

    editLink: {
      pattern: 'https://github.com/frank-1150/learning-ai-technical/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'Released under the MIT License',
      copyright: `Copyright © 2026-present`
    },

    outline: {
      level: [2, 3]
    }
  }
})

function nav(): DefaultTheme.NavItem[] {
  return [
    { text: 'Home', link: '/en/' },
    {
      text: 'AI Applications',
      items: [
        { text: 'RAG', link: '/en/ai-applications/rag/' },
        { text: 'Agents', link: '/en/ai-applications/agents/' },
        { text: 'Prompt Engineering', link: '/en/ai-applications/prompt-engineering/' }
      ]
    },
    {
      text: 'Machine Learning',
      items: [
        { text: 'Neural Networks', link: '/en/machine-learning/neural-networks/' },
        { text: 'Inference & Hardware', link: '/en/machine-learning/inference/nvidia-vera-rubin-lpx' }
      ]
    }
  ]
}

function sidebar(): DefaultTheme.Sidebar {
  return {
    '/en/ai-applications/': [
      {
        text: 'AI Applications',
        items: [
          { text: 'Overview', link: '/en/ai-applications/' },
          {
            text: 'RAG (Retrieval-Augmented Generation)',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/en/ai-applications/rag/' }
            ]
          },
          {
            text: 'Agents',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/en/ai-applications/agents/' }
            ]
          },
          {
            text: 'Prompt Engineering',
            collapsed: true,
            items: [
              { text: 'Introduction', link: '/en/ai-applications/prompt-engineering/' }
            ]
          }
        ]
      }
    ],
    '/en/machine-learning/': [
      {
        text: 'Machine Learning',
        items: [
          { text: 'Overview', link: '/en/machine-learning/' },
          {
            text: 'Inference & Hardware',
            collapsed: false,
            items: [
              { text: 'NVIDIA Vera Rubin + LPX', link: '/en/machine-learning/inference/nvidia-vera-rubin-lpx' }
            ]
          },
          {
            text: 'Neural Networks',
            collapsed: true,
            items: [
              { text: 'Introduction', link: '/en/machine-learning/neural-networks/' }
            ]
          }
        ]
      }
    ]
  }
}
