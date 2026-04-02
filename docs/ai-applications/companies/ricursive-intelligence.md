---
date: "2026-04-02"
title: "Ricursive Intelligence: AI for Chip Design & Chip Design for AI"
description: "Notes from Lightspeed VC's interview with Ricursive Intelligence founders — the AlphaChip inventors' story, three-phase roadmap, and why chip engineers are the world's most expensive architects"
tags: [chip-design, ai-hardware, startup, lightspeed, ricursive]
---

# Ricursive Intelligence: AI for Chip Design & Chip Design for AI

**Source**: Lightspeed Venture Partners interview series (YouTube)

AI is Designing the Chips that Power AI | Anna Goldie and Azalia Mirhoseini, Ricursive
https://youtu.be/VVwCyloDcL4?si=uTuBDb_6LoTc51rI
![](images/Pasted%20image%2020260402115216.png)

---

## Company Overview

**Note**: The company's official name is **Ricursive Intelligence**, not Recursive — the spelling is intentional, with the letter "i" emphasizing that Intelligence is embedded into the recursive loop itself.

**Founders**:
- **Anna Goldie** (CEO): Co-inventor of AlphaChip, former research scientist at Google Brain, early employee at Anthropic
- **Azalia Mirhoseini** (CTO): Co-inventor of AlphaChip, followed the same Google Brain → Anthropic → Google path

**AlphaChip Background**: The two co-founders used deep reinforcement learning at Google Brain to crack chip physical design (floor planning) — the most labor-intensive and hardest-to-automate step in chip design. AlphaChip is already commercially deployed, powering **four consecutive generations** of Google TPU designs (including the chips running Gemini, Imagen, and more), and has been adopted externally by semiconductor companies like MediaTek.

---

## Funding

| Round | Amount | Lead | Date |
|-------|--------|------|------|
| Seed | $35M | Sequoia Capital | Dec 2025 |
| Series A | **$300M** | **Lightspeed Venture Partners** | Jan 2026 |

Series A valuation: **$4 billion**, completed just **two months** after the company officially launched. Other participants: DST Global, NVentures (NVIDIA's VC arm), Felicis Ventures, Radical AI.

**Lightspeed's investment thesis** (distilled from their blog post): ① A founding team with production-scale real-world validation; ② The sheer scale of the semiconductor design market; ③ World-class engineering talent assembled from Google DeepMind, NVIDIA, Apple, and Cadence.

---

## Brilliant Interview Technique

The host noticed that the two founders kept **completing each other's sentences** — in perfect sync. The host then used this as a setup to ask: **"Why are you two so in sync?"**

This opened up the most fascinating chapter of their career story, immediately lifting the mood and energy of the entire conversation:

- Joined Google Brain on the same day
- Left Google Brain for Anthropic on the same day
- Left Anthropic and returned to Google on the same day, then left Google again on the same day

This wasn't planned — it's two people whose instincts about timing and career momentum are genuinely aligned. The interview demonstrated their chemistry as co-founders far more vividly than simply asking "how did you two meet?"

Anna as CEO showed an incredible level of focus throughout the interview — staring at the host without blinking for 60 seconds at a time, reminding me of those hyper-focused students in high school class.

---

## Key Insights

**The essential direction of AI: distill, accelerate, and automate the most valuable work**

The most valuable work isn't necessarily the most common work — chip engineers are the world's most expensive architects, and their time is being spent on tasks that don't require human intelligence to complete. This is the fundamental logic behind Ricursive's existence.

**Vibe coding expanded the entire market**

Previously, only about 25 million people could write software. With AI coding tools, at least 100 million people can now build software — the ceiling for coding tools has been dramatically raised. Anna and Azalia used this analogy to explain Ricursive's market logic: they're not competing for share in an existing market, they're creating a much larger new one.

**Google's internal repos are visible to all employees**

During their time at Google, both founders could access virtually all internal code repositories — that's Google's engineering culture, fully open to internal employees. For the training data quality of chip design AI, this represents an extremely rare unfair advantage. OMG.

**Amazing validation**

AlphaChip isn't a demo or a paper — it has been part of Google TPU's most critical production environment for four consecutive generations.

---

## Three-Phase Roadmap

### Phase 1: Accelerate existing chip design workflows

Compress tasks like floor planning and physical design from months to hours. Core customers are companies that already have chip design capabilities — help them go faster and cheaper.

### Phase 2: Chip design platform — enabling more companies to own their own chips

Today, a vast number of AI application companies run on general-purpose CPUs/GPUs because the barrier to custom chips is too high — you need the world's most expensive engineering team.

Ricursive's Phase 2 is to offer **design-less** (analogous to TSMC's fabless model): any company running its software at scale deserves custom silicon tailored to its needs. Clients focus on the application layer; Ricursive handles the silicon design underneath.

> "Any company that runs their software at scale properly deserves their own customized silicon."

### Phase 3: AI designing chips for AI — recursive self-improvement

Train better chip design methods with reinforcement learning → use those methods to build more powerful chips → use those chips to run better AI → iterate. This is what the name "Ricursive" truly points to.

This feels like a **democratized version of Jensen Huang's extreme co-design** vision: software and hardware shouldn't be optimized separately. Most software today — including some of the world's best AI models — is trapped on general-purpose hardware, unable to reach its true performance ceiling.

---

## Constructive Disruption

The host noted: during his research, he found that Ricursive's market isn't just the "semiconductor design tools" market (traditional EDA software — the Cadence/Synopsys world).

The real market cost breakdown is: **2/3 of the money goes to engineering teams, 1/3 to tools**. Ricursive attacks both simultaneously — replacing not just tools, but labor. The harder problem isn't even money — it's that **world-class chip engineers simply don't exist in sufficient numbers**, and every company globally is competing for the same talent pool. This echoes our first key insight: Ricursive doesn't just replace labor, it supplements the engineering expertise that some companies and markets simply cannot hire. It distills and concentrates chip design engineers' methods and experience for solving specific sub-problems and vertical challenges, then applies them at scale.

This is the real structural opportunity: it's not just about offering a better tool — it's about expanding the user base. Companies that couldn't build chips before can now build chips. Companies that couldn't hire engineers can now use Ricursive's solution. The market itself gets bigger.
