---
date: 2026-03-27
title: "vLLM & PagedAttention: Memory Revolution in LLM Inference"
description: Deep dive into the vLLM paper, understanding how PagedAttention uses OS virtual memory ideas to solve KV Cache memory management in LLM inference
tags: [vllm, pagedattention, inference, kv-cache, memory-management]
---

# vLLM & PagedAttention: Memory Revolution in LLM Inference

> This article is based on [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP '23, UC Berkeley).

::: info Chinese Version Available
The full article is currently available in Chinese: [vLLM 与 PagedAttention：LLM 推理的内存革命](/zh/machine-learning/inference/vllm-pagedattention)

English translation coming soon.
:::
