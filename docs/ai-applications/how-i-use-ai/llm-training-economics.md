---
date: 2026-05-28
title: "Packing Expensive Expertise Into a Model: The Economics Behind LLM Training"
description: Starting from a real NL2SQL project, thinking through what you should build yourself vs. what you should let the model labs build — the essence of a model company is wholesaling expensive expertise via fixed-cost training and amortizing it through mass distribution
tags: [llm, economics, training, infrastructure, business-model]
---

# Packing Expensive Expertise Into a Model: The Economics Behind LLM Training

Working on a recent Natural Language to SQL project forced me to think clearly about a question I'd been carrying for a while:

**As a product/business team, what should we build ourselves, and what should we wait for the model labs to build?**

It sounds abstract, but it lands on very concrete engineering decisions — and every choice has an economic structure underneath it.

## 1. Starting Point: A Private Model That Didn't Work

The original plan was simple: wire a SQL tool to an LLM so it could issue queries based on user questions, return results, and proactively explore the data for business insights.

For privacy reasons, our first version aimed at open-source models we could self-host. We tried Qwen3 and Qwen3-Coder. They ran — but the experience was rough:

- **Too few reasoning steps**: anything moderately complex got abandoned mid-task
- **Hallucinations once context grew**: after a few tool calls it started inventing table and column names
- **Unstable tool calling**: argument formats came out right sometimes, wrong other times, retry cost was high

We swapped the same tool stack onto Claude and it became immediately usable.

I had two paths in front of me:
1. **Patch it ourselves**: fine-tune, prompt-engineer, and scaffold the open model until it works
2. **Switch models**: just use a frontier model and let the open ones naturally catch up

My call was: **for this kind of problem, switch — don't patch.**

## 2. Why "Patching It Yourself" Is Almost Always a Losing Bet

The reason is straightforward. The capability you'd be trying to patch in — long tool-call chains, stable reasoning depth, low-hallucination context management — isn't your moat. It's **a general capability everyone wants**.

If you spend three months getting Qwen3 to "barely usable," half a year later Qwen4, Claude 5, or GPT-6 ships and your three months goes to zero. Worse, **the frontier labs are solving the same problem with budgets several orders of magnitude larger than yours** — and they're solving it better.

Andrej Karpathy made an observation in a recent podcast that stuck with me: compared to frontier models from 18 months ago, today's models are **vastly better at writing code, but barely any better at telling jokes**.

What does that tell us? It tells us the labs simply haven't invested in optimizing "jokes." **Model capability doesn't progress uniformly — it advances along the dimensions the labs choose to push.**

Their process for closing a capability gap is pretty standard:

1. Pick a target task (code, tool calling, long-context reasoning)
2. Collect enough samples and demonstrations
3. Push it through pre-training + post-training (SFT, RLHF/RLAIF)

If what you need happens to fall on a direction the labs **will** push — long-context tool use is a perfect example — you should **wait**, not race them.

## 3. So What Should You Build Yourself?

The flip side: here's what the frontier labs **won't** prioritize, and these are exactly your moat:

- Your company's schema, naming conventions, internal terminology
- Your permission model and sensitive-data boundaries
- Your customers' actual query patterns and high-frequency questions
- Your proprietary data and the domain knowledge surrounding it

None of these have a "general version." No matter how strong the frontier model gets, it has to go through **your prompts, your RAG, your tool definitions** to use any of this. That's where you should be spending people and time.

In one line: **wait on general capabilities; build vertical knowledge yourself.**

## 4. The Model Company Business: A High-Leverage Fixed Cost

Flip the lens around and you see what the model labs actually are, as businesses.

When Anthropic or OpenAI does a targeted training run — say, pushing code ability up a notch — it might burn tens of millions to hundreds of millions of dollars. But after training, that model gets **distributed to the entire world**, consumed by millions of developers across billions of calls.

**That one-time enormous cost, amortized across that much usage, becomes an extraordinarily high-leverage business.**

The logic is nearly identical to a search engine: Google builds an index once at huge compute cost, then amortizes it across hundreds of billions of queries. Both are **high fixed investment + massive distribution → marginal cost approaching zero**.

The headcount confirms this. A model lab's core team is often only a few dozen people, including a handful of top researchers, plus money paid to data-labeling firms. **You buy expensive human expertise, distill it into a model, then sell that expertise back to the world by the token.**

From this angle, **a model company is essentially a wholesaler of expertise**: it packs the world's most expensive professional capabilities into a model, then retails those capabilities through distribution.

## 5. The Vertical Opportunity: Ricursive Intelligence

Does that mean only the frontier labs can play this game? Not at all.

I wrote earlier about a company called [Ricursive Intelligence](../companies/ricursive-intelligence) that focuses on AI for chip design. Their thesis is to take the **high-barrier capability of chip design** and verticalize it into an AI product.

Why can Ricursive do this when Anthropic won't prioritize it?

- Chip design data is scarce, the domain expertise extreme — you need people with AlphaChip-level backgrounds to even enter the field
- It's a vertical market: even if Claude nailed it, it wouldn't translate into mass-user advantage
- Conversely, very few companies in the world have this capability or can assemble a team focused on it

That's exactly why Lightspeed funded them — **they're solving a vertical problem the frontier labs won't prioritize but that carries enormous commercial value.**

The vertical company's business logic is structurally identical to the frontier lab's: **a heavy upfront fixed cost to pack expensive expertise into a model, then recoup through distribution.** Just narrower market, higher price per customer.

## 6. One More Abstraction: Engineers Are All "Real Estate Designers"

There was a line in the Ricursive interview that hit me: **a chip designer is essentially the world's most expensive "real estate designer" — doing precision design on a microscopic plot of land; the fab is the construction crew that puts it in the ground.**

The metaphor scales out across every knowledge-dense industry:

- A model training team is the "designer": converting millions to hundreds of millions of dollars of fixed cost into a compact artifact that holds the world's most expensive expertise
- Inference infrastructure is the "construction crew": responsible for distributing and serving the artifact
- Every model call, every token billed, is recouping design cost and harvesting profit
- **Once paid back, the marginal cost of further distribution approaches zero — the model is just being copied**

It's a classic "heavy investment up front, light distribution after" business. Once you cross break-even, the leverage starts to release.

## 7. The Real Marginal Cost Lives in AI Infrastructure

But here's the catch: while copying the model is essentially free, **the real marginal cost lives in cloud services and AI infrastructure.**

Model companies are the **fixed-cost side** of this industry — R&D is one-time, distribution is free.
AI infrastructure (GPUs, inference clusters, networking, power) is the **variable-cost side** — every single intelligent call costs real physical resources.

That's why AI infrastructure attracts such large capital and is expected to return so much: **it corresponds to the water, electricity, and gas of every act of intelligence consumption in the AI era.** Every additional agent call, every additional model distribution, adds money to the infrastructure side of the ledger.

Model companies and infrastructure companies together form the cost structure of the AI industry — one side **amortizing past fixed investment**, the other side **absorbing present variable consumption**.

## Closing

Writing this through, I realized the underlying question of "what should I build vs. what should I wait for" is really asking: **which side of the cost structure do you want your work to sit on?**

- If you're building a general capability, you're racing **the world's fixed cost** — a race you can't win. Wait.
- If you're building vertical knowledge, business moat, proprietary data — you're **constructing your own small fixed cost**, waiting for your own leverage to release.

The AI infrastructure side is a different story altogether: **it doesn't run on leverage, it runs on consumption. It's the utility bill of the AI era.**

Once you can see these three layers, every AI project decision in front of you gets a lot clearer.
