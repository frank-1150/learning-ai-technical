---
title: Machine Learning
description: Core ML concepts and practice including PyTorch, tensors, and neural networks
tags: [machine-learning, deep-learning]
---

# Machine Learning

This section covers core machine learning theory and engineering practice.

## Topics

- [Neural Networks](./neural-networks/) — Neural network principles and implementation
- [Build GPT from Scratch (Karpathy)](./build-gpt-karpathy/) — Andrej Karpathy's walkthrough from Bigram to full Transformer
- [Inference & Hardware](./inference/nvidia-vera-rubin-lpx) — GPU/LPU cooperative inference, Roofline Model, vLLM & PagedAttention
- [CS336 Notes](./cs336/) — Stanford "Language Modeling from Scratch" assignment write-ups

## Latest inference retrospective

- [5.3× Serving Throughput for Qwen3.6-35B-A3B on a Single H20](./inference/qwen36-vllm-throughput) — An end-to-end measurement and diagnosis spanning prefix caching, FP8, scheduler step size, and kernel launch behavior
