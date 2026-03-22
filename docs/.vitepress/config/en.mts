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
        { text: 'Agent Principles & Control', link: '/en/ai-applications/agents/' },
        { text: 'How I Use AI', link: '/en/ai-applications/how-i-use-ai/' }
      ]
    },
    {
      text: 'Machine Learning',
      items: [
        { text: 'Neural Networks', link: '/en/machine-learning/neural-networks/' },
        { text: "Let's Build GPT", link: '/en/machine-learning/build-gpt-karpathy/' },
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
            text: 'Agent Principles & Control',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/en/ai-applications/agents/' },
              { text: 'The Core of Agents: Two Loops', link: '/en/ai-applications/agents/agent-loop' }
            ]
          },
          {
            text: 'How I Use AI',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/en/ai-applications/how-i-use-ai/' }
            ]
          },
          {
            text: 'AI Industry Views',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/en/ai-applications/industry-views/' }
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
            text: 'Neural Networks',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/en/machine-learning/neural-networks/' },
              { text: 'ChatGPT Overview (3Blue1Brown)', link: '/en/machine-learning/neural-networks/chatgpt-overview-3blue1brown' }
            ]
          },
          {
            text: "Let's Build GPT (Karpathy)",
            collapsed: false,
            items: [
              { text: 'Video Overview', link: '/en/machine-learning/build-gpt-karpathy/' },
              { text: 'PyTorch Basics', link: '/en/machine-learning/build-gpt-karpathy/pytorch-basics' }
            ]
          },
          {
            text: 'Inference & Hardware',
            collapsed: false,
            items: [
              { text: 'NVIDIA Vera Rubin + LPX', link: '/en/machine-learning/inference/nvidia-vera-rubin-lpx' }
            ]
          }
        ]
      }
    ]
  }
}
