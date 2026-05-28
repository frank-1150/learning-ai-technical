---
date: 2026-04-16
title: "NVIDIA Rubin R200: The Flagship GPU for the Inference Era"
description: A deep dive into the Rubin R200 GPU itself — 288GB HBM4, 22TB/s bandwidth, 50 PFLOPS NVFP4, NVLink 6, and the 10x MoE inference cost advantage over Blackwell — following the Vera Rubin + LPX architecture article
tags: [nvidia, rubin, r200, hardware, inference, hbm4, nvfp4]
---

# NVIDIA Rubin R200: The Flagship GPU for the Inference Era

> In [Vera Rubin + LPX architecture](./nvidia-vera-rubin-lpx.md), I wrote about why NVIDIA is moving FFN onto the LPU. This article continues by taking apart the main character itself — the **Rubin R200 GPU** — and explaining how it drives down MoE inference token cost by another order of magnitude.

## Why this deserves its own article today

Rubin was officially announced at GTC 2025, but it didn't enter full production until **CES 2026 (January)**, with the **first customer deliveries in August 2026**. This makes 2026 the transition year of "H100 → Blackwell → Rubin" all in service together — and everyone doing inference architecture selection is asking the same question:

> **How much can Rubin actually cut my token cost? When is it worth upgrading from Blackwell?**

NVIDIA's official numbers are **10x for MoE models, 2~3x for dense models**. This article breaks down the hardware spec by spec to show where the 10x comes from.

## Chip level: the key leap from Blackwell to Rubin

Rubin R200 is built on TSMC 3nm, with **two compute dies + two IO dies** packaged on a 4x reticle CoWoS-L substrate, for a total of **336B transistors** (1.6x Blackwell's 208B).

### Core spec comparison

| Metric | Rubin R200 | Blackwell B200 | H100 | Multiplier |
|---|---|---|---|---|
| Transistors | 336B | 208B | 80B | 1.6x vs B200 |
| SMs | 224 | 144 | 132 | 1.55x |
| **HBM capacity** | **288 GB HBM4** | 192 GB HBM3e | 80 GB | 1.5x |
| **HBM bandwidth** | **22 TB/s** | 8 TB/s | 3.35 TB/s | **2.75x** |
| **NVFP4 inference** | **50 PFLOPS** | ~10 PFLOPS | N/A | **5x** |
| NVFP4 training | 35 PFLOPS | ~10 PFLOPS | N/A | 3.5x |
| FP32 vector | 130 TFLOPS | 80 TFLOPS | 67 TFLOPS | 1.6x |
| FP64 matrix | 200 TFLOPS | 150 TFLOPS | 60 TFLOPS | 1.3x |
| TDP | 1800~2300W | 1000W | 700W | 2x |
| Cooling | **100% liquid (mandatory)** | Hybrid | Air | — |

## HBM4: bandwidth is the bloodstream of inference

Rubin's most important change isn't compute — it's **bandwidth**.

### How does HBM push bandwidth up?

A regular DDR5 DIMM has only a 64-bit-wide interface — a CPU-era convention, because the CPU access pattern is "small and random." The GPU access pattern is the opposite: **massive and regular**, so making the bus wider is the only way out.

HBM (High Bandwidth Memory) does three things to push bandwidth up:

**1. 3D stacking — the physical layer**

HBM stacks multiple DRAM dies vertically and connects them electrically through **TSVs (Through-Silicon Vias)** into one unit. An HBM4 stack contains **12~16 DRAM dies**, giving over 10x the effective circuit area of a traditional DIMM.

**2. Ultra-wide bus — the interface layer**

| Memory type | Interface width | Access mechanism |
|---|---|---|
| DDR5 (CPU) | 64 bit/channel | Long PCB traces |
| GDDR7 (consumer GPU) | 256~384 bit | High-speed differential serial |
| HBM3e | **1024 bit/stack** | Silicon interposer micron-scale bumps |
| **HBM4** | **2048 bit/stack** | CoWoS-L substrate |

There are thousands of micron-scale solder bumps between HBM and GPU die — a precision only achievable with advanced packaging like CoWoS, impossible to do on standard PCB. HBM4 doubles the bump density again.

**3. Pin rate — the signal layer**

HBM3e runs each pin at about **9.6 Gbps**, and HBM4 stays at **8~10 Gbps** per pin but doubles the pin count.

Combined, single-stack bandwidth:

$$
\text{Bandwidth} = \frac{\text{bus width} \times \text{pin rate}}{8 \text{ bit/byte}} = \frac{2048 \times 10 \text{ Gbps}}{8} \approx 2.75 \text{ TB/s}
$$

Eight stacks in parallel: **22 TB/s**.

### Why not just make DDR faster?

Electrically, higher DRAM clock means worse signal integrity and higher power — a hard physical constraint of $P \propto f \cdot V^2$. Instead of pushing every wire to its limit, you run thousands of wires in parallel. That's HBM's core philosophy: **trade parallelism for frequency**.

The cost of HBM bandwidth is **extremely high packaging cost, mandatory placement next to the GPU, and limited capacity** — which is why it will only ever live on data center GPUs and AI accelerators, never trickling down to consumers.

### Roofline view: why bandwidth directly equals inference performance

Arithmetic intensity for the Decode phase (see [Arithmetic Intensity](./nvidia-vera-rubin-lpx.md#什么是-arithmetic-intensity)):

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

For MoE Decode, this is typically **1 FLOP/Byte**, meaning each byte of weights pulled in supports only one float op. The real GPU ceiling here:

- H100: 3.35 TB/s × 1 = **3.35 TFLOPS** effective ceiling (theoretical 1 PFLOPS FP16 wasted 99.7%)
- B200: 8 TB/s × 1 = **8 TFLOPS**
- R200: 22 TB/s × 1 = **22 TFLOPS**

**In the memory-bound regime, bandwidth IS compute.** That's the fundamental reason Rubin "looks like 5x compute, but only 3x actual inference gain" — because most of the gain comes from bandwidth, not FLOPS.

## Precision and quantization: from FP32 to NVFP4

This section is the most counterintuitive for engineers without an ML background — to understand why NVFP4 doubles compute, you have to take the word "precision" apart first.

### How regular programming float/int maps to GPU precision

In C/Java/Python, `float` is almost always **IEEE 754 FP32**: 32 bits, structured as "1 sign + 8 exponent + 23 mantissa." `double` is FP64. `int` is integer, no exponent, **uniformly distributed** on the number line.

LLM weights and activations are floating point under the hood, but FP32 is a luxury for deep learning — a 120B model is 480 GB of weights alone. The last decade of hardware has been evolving toward **shorter float formats**:

| Format | Total bits | Layout (sign/exp/mantissa) | Dynamic range | Relative precision | Use |
|---|---|---|---|---|---|
| FP64 | 64 | 1/11/52 | Huge | Highest | Scientific computing |
| FP32 | 32 | 1/8/23 | Wide | High | Legacy training baseline |
| TF32 | 19 | 1/8/10 | Same as FP32 | Medium | Introduced in Ampere, training matmul |
| FP16 | 16 | 1/5/10 | Limited | Medium | Mixed-precision training |
| BF16 | 16 | 1/8/7 | Same as FP32 | Low | Default for training (pushed by Google) |
| FP8 E4M3 | 8 | 1/4/3 | Small | Very low | Forward pass |
| FP8 E5M2 | 8 | 1/5/2 | Medium | Extremely low | Gradient |
| **NVFP4** | **4** | **1/2/1 + block scale** | Extended by scale | Extremely low | Inference |
| INT8 | 8 | Integer | ±127 | Uniform | Legacy quantization |
| INT4 | 4 | Integer | ±7 | Uniform | Legacy quantization |

A few key observations:

- **FP16 vs BF16**: same 16 bits, but FP16 has higher precision and narrower range (overflows at ±65504); BF16 trades mantissa for FP32-equivalent exponent range, making training more stable. Google's TPU pushed it first, now everyone supports it.
- **Two FP8 variants**: E4M3 is precision-first for forward pass, E5M2 is range-first for gradients.
- **FP vs INT**: at the same N bits, FP wins when values **span very different magnitudes** (the exponent gives a log scale), while INT is cheaper when distribution is uniform.
- **NVFP4**: at only 4 bits, both mantissa and exponent are tiny — almost incapable of carrying information by themselves. Needs **a shared scale per small block** to work at all.

### How does quantization actually work?

Quantization compresses high-precision values like FP32/FP16 into low-precision formats like INT8/FP8/FP4 for storage and computation. The core formula:

$$
x_{q} = \text{round}\left(\frac{x}{\text{scale}}\right), \quad x \approx x_{q} \times \text{scale}
$$

The scale is a scalar that maps the actual range of input values onto the target format's representable range. For example, INT8 can represent ±127; if your weight range is ±6, then scale = 6/127 ≈ 0.047.

**Three quantization granularities**:

| Granularity | # of scales | Precision | Storage cost |
|---|---|---|---|
| Per-tensor | 1 | Worst — entire tensor shares one | Near zero |
| Per-channel | One per row/column | Good — output dimension independent | Medium |
| **Per-block (microscaling)** | **One per 16~32 elements** | **Best — locally adaptive** | Needs to store scales |

NVFP4 uses **per-block microscaling**: **16 FP4 values share one FP8 scale**. This is NVIDIA's variant of the 2025 OCP Microscaling (MX) formats standard. The core innovation is keeping the scale "local enough" to absorb local extremes, with model precision loss typically < 1%.

<HtmlVisualization
  src="/machine-learning/inference/visualizations/precision-formats.html"
  height="560px"
  title="Precision format bit layouts: enter a number and see how each format quantizes it"
/>

### 3rd-gen Transformer Engine: hardware-automated quantization

Traditional quantization workflow is "train → run an offline quantization pass → get a quantized inference model." Transformer Engine bakes this **into the Tensor Core itself**:

- During every matmul, hardware **tracks activation dynamic range in real time**
- It **picks a scale and generates NVFP4 inputs automatically** based on those stats
- Compute is done in NVFP4, accumulation in FP32 (to avoid accumulation error)
- The developer doesn't touch model code

This is the first time LLM teams can **train in BF16 and let inference automatically downgrade to NVFP4** with near-free precision loss.

### Why does compute double as precision halves?

A Tensor Core is a fixed-area matrix multiply unit. For the same transistor count:

- A unit doing 1 FP16 multiply can be split into **2 FP8 multiply units** (half the bit width)
- Or **4 FP4 multiply units** (half again)
- Storage/transfer scales the same way — one FP16's worth of space holds 4 FP4s

So Rubin's FP4 compute (50 PFLOPS) ≈ 2x its FP8 (25 PFLOPS) ≈ 4x its FP16. **Precision and compute are strictly inverse** — that's why hardware vendors keep chasing lower precision. But this only pays off for **tasks that tolerate precision loss** (inference, training that can absorb error); high-precision workloads (scientific computing, FP64 matmul) can't cash in this dividend.

## Vera CPU: the first CPU with native FP8 support

The **Vera CPU** that ships with Rubin is NVIDIA's own ARM processor:

- 88 Olympus cores, Armv9.2, with spatial multithreading (SMT) giving **176 hardware threads**
- **First CPU with native FP8 precision support** (6 × 128-bit SVE2 SIMD per core)
- 1.5 TB LPDDR5X, 1.2 TB/s bandwidth
- 227B transistors (monolithic, not chiplet)
- Connects to Rubin via NVLink-C2C at **1.8 TB/s** (7x faster than PCIe Gen 6)

### What does "native FP8 support" actually mean?

This phrase came up over and over in the keynote, and it's easy to read as marketing. The strict definition of **native support** is: **the CPU instruction set has instructions that operate directly on FP8 operands**, so FP8 add/multiply/FMA (fused multiply-add) can run inside SIMD registers.

What happens without native support? You read FP8 from memory, **convert to FP32 or FP16 first**, run scalar/SIMD pipelines, then convert back to FP8 to store. Three problems:

1. **Conversion itself burns cycles** — unpack + bit manipulation, often 3~5 extra instructions
2. **Intermediate results occupy wider registers**, dropping SIMD parallelism (a 128-bit register goes from 16 parallel FP8s down to 4 FP32s)
3. **No direct SIMD FMA** — accumulation needs an extra store/load

Vera's every SVE2 unit can **process 16 FP8 elements in parallel** inside a 128-bit register with FMA. The whole CPU — 88 cores × 6 SVE2 units × GHz-class clock — has enough headroom to handle tokenization, embedding lookups, quantization-scale computation, RL reward-model scoring, and other auxiliary work.

This solves a long-standing pain point: when the GPU is doing FP8 compute and the CPU can't keep up, **data has to be converted back to FP32 for the CPU, then back to FP8 to ship back to the GPU**, and that round trip eats into the GPU's effective throughput. Vera's native FP8 turns the CPU into a "same-language coprocessor" for the GPU, so the pipeline never stalls.

Vera's role is the GPU's **data preprocessing butler** — tokenization, KV cache management, scheduling decisions all happen here, freeing the GPU to focus 100% on matrix compute.

## Vera Rubin NVL72: the rack-scale combat unit

A single R200 is powerful, but what NVIDIA actually sells is the **whole rack**: **Vera Rubin NVL72**.

| Metric | Vera Rubin NVL72 | Grace Blackwell NVL72 | Gain |
|---|---|---|---|
| # GPUs | 72 | 72 | — |
| # CPUs | 36 | 36 | — |
| NVFP4 inference | **3.6 EFLOPS** | ~720 PFLOPS | 5x |
| Total HBM | 20.7 TB | 13.5 TB | 1.5x |
| HBM aggregate bandwidth | **1.6 PB/s** | 576 TB/s | 2.8x |
| NVLink aggregate bandwidth | **260 TB/s** | 130 TB/s | 2x |
| System memory | 54 TB LPDDR5X | 17 TB | 3.2x |
| Power delivery | **800V DC** | Conventional AC | — |
| Cooling | Liquid (45°C inlet) | Liquid | — |
| Total power | >250 kW | ~130 kW | — |
| Per-rack cost | **$3.5~4M** | $3.35M | 1.2x premium |

Mechanical dimensions are **identical to Blackwell NVL72** — a deliberate design choice from NVIDIA so customers can do **drop-in replacement** without rewiring their data centers.

## Three-tier interconnect: Scale-up / Scale-out / Scale-across

AI workloads keep getting bigger, and simply "stacking more GPUs" isn't enough — different **communication granularities** need different interconnect layers. NVIDIA's 2026 product lineup is organized in three tiers:

| Tier | Scope | Core requirement | NVIDIA solution | Typical bandwidth | Typical latency |
|---|---|---|---|---|---|
| **Scale-up** | Within a rack (72 GPUs) | Shared memory semantics | NVLink 6 + NVL72 | 3.6 TB/s per GPU | ~100 ns |
| **Scale-out** | Within a DC (thousands~tens of thousands GPUs) | High-bandwidth IP network | Spectrum-X / Quantum-X + BlueField-4 | 800 Gbps per link | μs class |
| **Scale-across** | Across DCs (global) | Geo-scale deployment | Spectrum-XGS | Tens~hundreds of Gbps | ms class |

Top to bottom: **bandwidth drops, latency rises, distance grows, per-bit cost falls**. Each tier matches a different class of workload.

<HtmlVisualization
  src="/machine-learning/inference/visualizations/interconnect-hierarchy.html"
  height="520px"
  title="Click each layer for details: NVIDIA's three-tier interconnect, products, bandwidth, and workloads"
/>

### Scale-up: NVLink 6 (tightly coupled shared-memory domain)

The goal of scale-up is to make a group of GPUs **act as one large GPU**: shared memory view, atomic operations, sub-microsecond latency.

| Technology | Rubin gen | Blackwell gen | Use |
|---|---|---|---|
| **NVLink 6** | 3.6 TB/s per GPU | 1.8 TB/s | GPU ↔ GPU |
| **NVLink Switch** | 260 TB/s aggregate / 72 GPUs | 130 TB/s | All-to-all inside a rack |
| **NVLink-C2C** | 1.8 TB/s | 900 GB/s | Vera CPU ↔ Rubin GPU coherent link |

**Key technology**: NVLink runs on NVIDIA's own **NVHS physical layer** (differential signaling + custom protocol), not PCIe and not Ethernet. Copper cables, distance limited to a few meters inside the rack.

**Matching product**: **Vera Rubin NVL72** (72 GPUs + 36 CPUs rack-scale combat unit).

**Suitable workloads**:
- **Tensor Parallel** — all-reduce across cards per-layer matmul, extremely frequent
- **Expert Parallel** — MoE all-to-all routing
- **Prefill node internals in P/D disaggregation** — single-request multi-card compute split
- **Tightly coupled single-model serving**

### Scale-out: Spectrum-X + Quantum-X + BlueField-4

Beyond the rack is scale-out range. The base layer is **Ethernet or InfiniBand** — still IP packet switching at heart, but heavily modified for **lossless, no-drop, low-latency** AI traffic.

| Component | Type | Spec | Role |
|---|---|---|---|
| **Spectrum-6 SN6800** | Ethernet ASIC | 409.6 Tb/s / 512 × 800 Gbps | Ethernet switching base |
| **Quantum-X800** | IB ASIC | 115 Tb/s | InfiniBand switching (HPC / large-scale training) |
| **ConnectX-9 NIC** | NIC | 800 Gbps | GPU egress |
| **BlueField-4 DPU** | Smart NIC | 800 Gbps + 64-core CPU + 128 GB RAM | Management / storage / security offload |

**Key technology — CPO (Co-Packaged Optics)**:

Traditional optical modules are **pluggable**; a single 800 Gbps link's optics burns 15~20W. Spectrum-6 packages the **optical engine and ASIC on the same substrate**, removing the back-and-forth electrical conversion between ASIC → SerDes → retimer → optical module, **cutting per-port power by 5x**. For million-GPU data centers, that's tens of megawatts saved — directly determining whether they can be built at all.

**BlueField-4's ICMS (Inference Context Memory Storage)**:

BlueField-4 packs 126B transistors + 64-core Grace CPU + 128 GB LPDDR5, basically a standalone DPU server. Its ICMS is built for **PB-scale KV cache** — allowing KV cache to persist on the DPU's local SSDs. This is the hardware-side answer to the same trend driving the layered KV cache in [Mooncake](./prefill-decode-disaggregation-mooncake.md).

**Suitable workloads**:
- **Data Parallel training** — gradient all-reduce across racks
- **Cross-node KV transfer in P/D disaggregation** (Mooncake Transfer Engine rides this tier)
- **RAG inference** — vector DB access
- **Multi-tenant inference serving** — DPU does isolation and scheduling

### Scale-across: Spectrum-XGS (across DCs)

This is a new category introduced in 2025. Drivers:

- A single DC's **power, land, and cooling** are all hitting physical limits (a few hundred MW is the ceiling)
- Million-GPU training jobs need **2~3 geographic sites** to find capacity
- Regulations require inference to happen **inside the user's country**
- Disaster recovery requires critical inference services to fail over across regions

The scale-across solution is **Spectrum-XGS**, supporting **Giga-Scale cross-site** topologies. Key technologies:

- **Large-latency-tolerant RDMA**: still congestion-free over tens of ms of distance
- **End-to-end telemetry + congestion-aware routing**: the scheduler senses path congestion in real time
- **Omniverse coordination**: digital twins for capacity planning and failure rehearsal

**Suitable workloads**:
- **Super-large foundation model training** (multi-DC deployments at OpenAI, xAI)
- **Global inference load balancing**
- **DR and regulatory isolation**

### How the three tiers relate

A typical hyperscale MoE training job uses all three tiers at once:
- **Scale-up**: each expert's matmul does tensor parallel inside an NVL72
- **Scale-out**: all-to-all between experts across racks rides Spectrum-X
- **Scale-across**: shards of the whole training job span DCs via Spectrum-XGS

NVIDIA's strategy is to **own all three tiers** — and this is its deepest moat: not every competitor can simultaneously build GPUs, switching ASICs, DPUs, and optical packaging.

## Real cost: why per-million-token can drop to a few cents

NVIDIA's headline benchmark is **Kimi-K2-Thinking MoE, 32K input / 8K output**:

| Hardware | Cost per million tokens (estimated) |
|---|---|
| Hopper H100 | ~$0.20 |
| Blackwell FP8 | ~$0.10 |
| **Rubin NVFP4 (MoE)** | **~$0.005~0.01** |
| Rubin NVFP4 (Dense) | ~$0.02~0.03 |

### The cost formula: factor-by-factor

Cost per token can be written as:

$$
\text{Cost/token} = \frac{\text{GPU-hours} \times \text{GPU unit price}}{\text{tokens generated}} \times \text{PUE}
$$

- The numerator is **hardware depreciation cost** (equivalent hourly rent of a Rubin)
- The denominator is **tokens produced per unit time**
- PUE is the **data center energy utilization correction factor**

Rubin's 10x cost reduction isn't a single optimization — it **explodes the denominator while simultaneously dropping the numerator and PUE**. Factor by factor:

### Factor 1: precision density (FP16 → NVFP4, 4x compute density)

Each Tensor Core's transistors can run **4x more multiplications in 4-bit than FP16**. Direct consequence:

**For Prefill (compute-bound): same GPU does 4x more tokens per unit time** → denominator ×4.

But the precision drop isn't free — it relies on 3rd-gen Transformer Engine + per-block microscaling to keep precision loss < 1%. **Software-emulated FP4 doesn't count**: you lose the compute advantage because FP4 still has to be unpacked to FP16 to be computed, missing the hardware dividend.

Dense models' Prefill realizes essentially the full 4x.

### Factor 2: memory bandwidth (8 TB/s → 22 TB/s, 2.75x)

Decode is memory-bound. Then:

$$
\text{Decode tokens per second} \approx \frac{\text{HBM bandwidth}}{\text{bytes of weights read per token}}
$$

Generating each Decode token requires pulling the entire activation-path weights once. Bandwidth × 2.75, so **the same GPU produces 2.75x more Decode tokens per unit time** → denominator × 2.75.

**This is where MoE and Dense diverge the most**:

- **Dense models**: every token reads the full 480 GB of weights; no matter how high bandwidth goes, you only get 2.75x.
- **MoE models**: every token only activates 10~15B parameters (~10% of total). Weight bytes per token are an order of magnitude smaller. Bandwidth utilization approaches 100%, the 2.75x is fully captured, with an additional multiplier from sparse activation on top.

That's why NVIDIA quotes 10x for MoE but only 2~3x for Dense — MoE Decode **was designed for high-bandwidth hardware**.

### Factor 3: PUE (1.5 → 1.1, saves 35% of DC electricity)

Data center **PUE (Power Usage Effectiveness)**:

$$
\text{PUE} = \frac{\text{Total DC power}}{\text{IT equipment power}}
$$

- Air-cooled DCs have PUE around **1.5**: every 1 kW of compute burns another 0.5 kW in cooling and power distribution.
- Liquid + 800V DC direct delivery drops PUE to **1.1**: only 0.1 kW extra burned.

Why can Rubin drive PUE this low?

1. **100% liquid cooling mandated**: liquid has 4000x the heat capacity of air, a step change in cooling efficiency.
2. **800V DC direct delivery**: skips multi-stage low-voltage AC → DC conversion. Each stage loses 2~5%; skipping 3~4 stages saves 10~15% of electricity directly.
3. **CPO optical modules cut power 5x**: at million-GPU scale, that's tens of MW saved on the network side.

PUE going from 1.5 → 1.1 means **35% better utilization of every kWh into the DC**, which divides the PUE factor at the tail of the cost formula by 1.35. This is the hard opex dividend.

### Factor 4: NVLink 6 (scale-up domain bandwidth doubled, higher GPU utilization)

Looks unrelated to token cost, but actually critical. Doubled NVLink bandwidth lets **larger models fit inside a single scale-up domain (NVL72)**.

In the past, 1T-class MoE had to do tensor parallel across racks. Cross-rack Spectrum-X has **more than 10x less** bandwidth than NVLink; communication cost drowns compute, producing massive **pipeline bubbles** (idle GPU time waiting for data). Typical outcome: 40~50% GPU utilization.

Rubin + NVL72 fits an entire 1T MoE inside the scale-up domain: communication rides NVLink, **utilization climbs to 80%+**.

This acts directly on the numerator: **the same GPU-hours produce 1.5~2x more actual work**.

### Factor 5: MoE sparse activation (software-hardware co-design dividend)

A 120B MoE only activates 10~15B params per token. Implications:

- **Memory demand = total parameters** (need to hold all experts; 288 GB HBM is just right)
- **Compute per token = activated parameters** (only the chosen expert runs)
- **Bandwidth per token = activated parameters** (only the chosen expert's weights are read)

Rubin's 288 GB HBM is exactly enough to hold a full 120B MoE without splitting across nodes. Meanwhile 22 TB/s makes reading the active expert's weights blazing fast. **MoE + Rubin is the sweet spot designed by architecture and hardware together** — not a coincidence.

### Putting it together: how 10x stacks up

These factors **don't just multiply** — they have to be Prefill/Decode-weighted first:

| Factor | Prefill gain | Decode gain (MoE) | Decode gain (Dense) |
|---|---|---|---|
| FP4 density | 4x | 1x (decode bound on bandwidth) | 1x |
| HBM4 bandwidth | ~1.5x | 2.75x | 2.75x |
| NVLink/utilization | 1.5x | 1.5x | 1.2x |
| PUE 1.5→1.1 | 1.35x | 1.35x | 1.35x |
| MoE sparsity | — | 2~3x | — |
| **Combined (geometric weight)** | **~6x** | **~8~10x** | **~2~3x** |

Weighted-averaged across Prefill and Decode (in the Kimi K2 scenario decode dominates), MoE's combined gain is **8~10x**, while Dense stops at 2~3x.

**So "10x cost reduction" is real for large-model MoE inference, but diluted for small/Dense models** — which is why the timing recommendation is "go Rubin today for long-context / MoE-heavy inference; stick with Blackwell for short prompts and small models." Rubin's architecture is a bet that MoE will be the 2026 mainstream.

## Timeline: 2026 to 2028

| Time | Milestone |
|---|---|
| 2026-01 (CES) | Rubin officially launches, full production |
| 2026-08 | First customer deliveries (Quanta confirmed) |
| 2026-Q4 | AWS / GCP / Azure / CoreWeave / Lambda / Nebius come online in succession |
| 2027-H2 | **Rubin Ultra**: 4 compute dies, 100 PFLOPS FP4, 1 TB HBM4e |
| 2028 | **Feynman architecture** (TSMC 1.6nm) |

NVIDIA officially commits to a "strict yearly cadence," with each generation delivering **3~5x inference gain + 2~3x training gain**.

## My take

Rubin isn't a "wow" architecture upgrade — it's a product that **pushes the Blackwell direction to its extreme**:

- **HBM4 takes bandwidth to 22 TB/s**, making memory-bound Decode performance directly proportional to bandwidth
- **NVFP4 doubles compute density**, letting Compute-bound Prefill cash in the process dividend
- **CPO and 800V DC** solve the power bottleneck for hyperscale deployments
- **NVL72 mechanical compatibility** minimizes upgrade friction

The real strategic value is that **NVIDIA has redefined "inference economics" with R200** — driving token cost from cents to fractions of a cent for the first time, making sustainable business models for AI applications viable.

If you're sizing inference infrastructure, my recommendations:

- **Go Rubin today**: long-context / MoE-heavy inference workloads
- **Stay on Blackwell**: short prompts, small models, power/price-sensitive
- **H100 retirement window**: 2026 Q4 ~ 2027 Q2, when Rubin supply is plentiful and prices stabilize

## References

- [NVIDIA Rubin at GTC 2026 Technical Breakdown (Barrack AI)](https://blog.barrack.ai/nvidia-rubin-specs-architecture-2026/)
- [NVIDIA Acqui-Hire of Groq for Rubin Platform](https://markets.financialcontent.com/stocks/article/tokenring-2026-1-21-nvidia-seals-20-billion-acqui-hire-of-groq-to-power-rubin-platform-and-shatter-the-ai-memory-wall)
- [AI Hardware Companies 2026 (Big Data Supply)](https://bigdatasupply.com/leading-ai-hardware-companies/)
- Related on this site: [Vera Rubin + LPX](./nvidia-vera-rubin-lpx.md), [Prefill/Decode Disaggregation and Mooncake](./prefill-decode-disaggregation-mooncake.md)
