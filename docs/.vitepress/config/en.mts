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
    { text: 'Home', link: '/' },
    {
      text: 'AI Applications',
      items: [
        { text: 'Agent Principles & Control', link: '/ai-applications/agents/' },
        { text: 'How I Use AI', link: '/ai-applications/how-i-use-ai/' },
        { text: 'AI Industry Views', link: '/ai-applications/industry-views/' }
      ]
    },
    {
      text: 'Machine Learning',
      items: [
        { text: 'Neural Networks', link: '/machine-learning/neural-networks/' },
        { text: "Let's Build GPT", link: '/machine-learning/build-gpt-karpathy/' },
        { text: 'Inference & Hardware', link: '/machine-learning/inference/nvidia-vera-rubin-lpx' }
      ]
    }
  ]
}

function sidebar(): DefaultTheme.Sidebar {
  return {
    '/ai-applications/': [
      {
        text: 'AI Applications',
        items: [
          { text: 'Overview', link: '/ai-applications/' },
          {
            text: 'Agent Principles & Control',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/ai-applications/agents/' },
              { text: 'The Core of Agents: Two Loops', link: '/ai-applications/agents/agent-loop' }
            ]
          },
          {
            text: 'How I Use AI',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/ai-applications/how-i-use-ai/' },
              { text: 'My AI Information Sources', link: '/ai-applications/how-i-use-ai/my-ai-information-source' },
              { text: 'In the Agent Era, the Real Demand May Be "Better at Execution"', link: '/ai-applications/how-i-use-ai/agent-scheduling-thoughts' }
            ]
          },
          {
            text: 'AI Industry Views',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/ai-applications/industry-views/' },
              {
                text: 'Industry Consensus',
                collapsed: true,
                items: [
                  { text: 'List', link: '/ai-applications/industry-views/consensus/' },
                  { text: '2026-03', link: '/ai-applications/industry-views/consensus/2026-03' }
                ]
              },
              {
                text: 'Industry Perspectives',
                collapsed: true,
                items: [
                  { text: 'List', link: '/ai-applications/industry-views/perspectives/' },
                  { text: '2026-03', link: '/ai-applications/industry-views/perspectives/2026-03' }
                ]
              },
              {
                text: 'Facts & Data',
                collapsed: true,
                items: [
                  { text: 'List', link: '/ai-applications/industry-views/facts/' },
                  { text: '2026-03', link: '/ai-applications/industry-views/facts/2026-03' }
                ]
              }
            ]
          }
        ]
      }
    ],
    '/machine-learning/': [
      {
        text: 'Machine Learning',
        items: [
          { text: 'Overview', link: '/machine-learning/' },
          {
            text: 'Neural Networks',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/machine-learning/neural-networks/' },
              { text: 'ChatGPT Overview (3Blue1Brown)', link: '/machine-learning/neural-networks/chatgpt-overview-3blue1brown' }
            ]
          },
          {
            text: "Let's Build GPT (Karpathy)",
            collapsed: false,
            items: [
              { text: 'Video Overview', link: '/machine-learning/build-gpt-karpathy/' },
              { text: 'PyTorch Basics', link: '/machine-learning/build-gpt-karpathy/pytorch-basics' }
            ]
          },
          {
            text: 'Inference & Hardware',
            collapsed: false,
            items: [
              { text: 'NVIDIA Vera Rubin + LPX', link: '/machine-learning/inference/nvidia-vera-rubin-lpx' }
            ]
          }
        ]
      }
    ]
  }
}
