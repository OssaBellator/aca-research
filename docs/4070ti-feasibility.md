# RTX 4070 Ti feasibility, ranking and resource envelope

## Hardware assumption and measured PC

The target device `PC` was probed via the Minimal MCP fleet on 2026-11-03: **RTX 4070 Ti, 12,282 MiB VRAM, NVIDIA driver 610.88, 31.81 GiB system RAM, Python 3.11, PyTorch not installed**. See [pc-hardware.md](pc-hardware.md). Cost and throughput figures below are **still unmeasured**; an inventory does not establish peak free GPU memory or model performance.

### Specification and capacity calculations

**Standard GeForce RTX 4070 Ti: 12 GB GDDR6X VRAM**, 7,680 CUDA cores. The distinct **RTX 4070 Ti SUPER: 16 GB**. Manufacturer: https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4070-family/ .

This document assumes one standard 12 GB card; host RAM, SSD, OS, CUDA/toolchain and available PCIe bandwidth are unknown until measured. For reproducibility save `nvidia-smi -q`, `python -c "import torch; print(torch.__version__,torch.cuda.is_available())"`, OS, RAM, storage and exact model configurations.

## Memory mathematics (illustrative, not a measured profile)

For conventional K and V tensors, without padding or quantization:
`KV_bytes = 2 * num_layers * num_kv_heads * head_dim * bytes_per_scalar * tokens`.

Examples with BF16, 2 bytes/scalar:

| Shape | KiB/token | 32,768 tokens | 131,072 tokens |
|---|---:|---:|---:|
| 24 layers, 8 KV heads, 64 dim | 48 KiB | 1.5 GiB | 6 GiB |
| 32 layers, 8 KV heads, 128 dim | 128 KiB | 4 GiB | 16 GiB |

The second example alone exceeds the card at 131,072 tokens **before** model weights and runtime overhead. A 1B parameter model in BF16 requires about 1.86 GiB of raw weights, plus buffers/activations; 4-bit weights are not literally 0.5 GB because metadata and temporary tensors also consume memory.

Do not assume a model supports arbitrarily long contexts just because the KV can be offloaded. Some attention implementations/OOM risks occur during prefill; kernels can impose hard limits.

## Ranked investigations — single card

Rank is qualitative: **research value × measurable differentiation × ability to run on 12 GB × speed to reproducible evidence**. Hardware hours are rough planning envelopes, **not benchmarked**.

| Order | Hypothesis | Local experiment | Expected effort | Outcome useful to researchers? | Main caveat |
|---:|---|---|---|---|---|
| **1** | **B02** predictive KV prefetch | replay attention-page traces; compare LRU/query-aware/predictor/oracle under *measured* or sensitivity-swept transfer latency | CPU: hours; optional GPU profiling: ~1–6 h | Yes, a reproducible latency/cost boundary or failure | trace quality; synthetic predictions overestimate gains |
| **2** | **C03** rare-fact retention | sequential episodes with storage ceiling; rare future-query surprise; compare LRU/LFU/reservoir/novelty | CPU: hours–days | Yes, if robust against different future-query distributions | policy could exploit artificial query generator |
| **3** | **B05** KV version safety | deliberately mismatch weights/positions/prefixes; validate fingerprints and mismatch detection | CPU/GPU: <~4 h | Useful engineering correctness test | not itself an architectural breakthrough |
| **4** | **B11** memory thrashing | alternating-topic page-cache stress, hysteresis and prediction | CPU: hours | Useful operating-regime study | no hardware timing unless calibrated |
| **5** | **A04** timing of stream exchange | tiny recurrent network on staged-constraint task; early/mid/late message ablations | GPU ~2–12 h after setup | High scientific value if communication causally helps | synthetic-task transfer weak |
| **6** | **A02/A07** budget router | mixed-difficulty symbolic tasks; distil oracle budgets into a cheap predictor | CPU or GPU ~2–10 h | Useful if beats fixed equal-cost allocation | costly teacher-label sweeps |
| **7** | **C02** exact archive + consolidation | long sequence, rare fact changes, limited index size; benchmark summaries | CPU + optional small model ~1–8 h | Useful for assistant memory systems | retrieval quality dominates |
| **8** | **B03/B09** sparse attention fidelity | small GQA model with query-aware page selection at 4–32k tokens | GPU ~4–20 h plus kernel work | Could reveal accurate, memory-saving regime | Quest already strong; implementation fairness |
| **9** | **A03** causal 2-stream benefit | recurrent latent network vs serial/equal FLOPs/voting with matched budgets | toy GPU ~8–40 h; LLM-scale much more | Highest ACA novelty if positive | smallest models may not learn useful streams |
| 10 | B06/B07 | real CPU spill vs recompute latency and p95 memory pressure | GPU ~4–20 h plus systems work | Useful if forced KV pressure | single PCIe GPU may lose badly |
| 11 | C04/C05 | sequential adapter updates + rehearsal controls | GPU many hours depending model/dataset | Interesting continual learning | slow, forgetting confounds, OOM |
| 12 | D01/D03/D04 | joint controllers and scale transfer | not credible as first local test | Only after earlier wins | architecture and compute too costly |

**Why B02 beats full ACA as a first project:** it tests one precise architectural idea with trace-driven replay and baselines, uses the exact memory-management motivation, and gives a falsifiable result without training a large model. The novel part must be carefully distinguished from already published predictive caching and Quest-style attention selection.

**Why A03 is the most valuable ambitious question:** communicating latent states might provide genuinely new reasoning quality; however, the hardest rival is simply generating two independent candidates and voting, or serially spending the same FLOPs.

## Run strategy on 12 GB

- Begin with independent **CPU simulations** for B02/C03/B05/B11. This does not require PyTorch.
- Use 0.1–0.5B scale or a small 0.5–1.5B open-weight model for traces and tiny adaptation only after profiling; model architecture, context length and license matter more than nominal parameter count.
- Keep training microbatch small, use BF16 where supported, gradient accumulation, activation checkpointing, and LoRA/adapters when weights are updated. Assume full training of an ambitious multi-stream LLM is **not** economical on one 4070 Ti.
- For full-attention comparisons beyond local VRAM, use offline precomputed reference traces, or shorter workloads with an exact full-context oracle; don't compare impossible dense GPU baselines as if they were run.
- Report GPU **elapsed hours**, not H100-equivalent hours. Do not claim equivalence to an H100 pilot from a FLOP ratio alone.
- CPU offload requires RAM; paging to SSD can dominate latency. Test pressure at 8k/16k/32k then scale until OOM or performance cliff (within model's supported context).
- CPU and GPU runtime budgets should be set from a first **smoke profile** rather than from this document's estimates.

## Decision rules

**Continue B02** only if predicted prefetch has positive net benefit vs tuned simple alternatives across at least one prespecified trace family and transfer regime, after counting predictor and transfer overhead. A negative result with precise boundary conditions is still reportable.

**Continue A03** only if communication improves held-out accuracy at matched compute vs strongest voting and serial-loop baselines, and inference-time message corruption causally harms the model.

**Scale any claim** only with second model/task distribution, 3+ independent seeds when training, bootstrap confidence intervals across test examples, and an untouched final test set.

## New model candidate: NVIDIA Dory 1.0 4B

[Dory assessment and pilot](dory-evaluation.md): its Q5_K_M 3.67 GB GGUF checkpoint may fit this PC for 4096-token loop-count inference. Its adjustable 1/2/4 recurrent passes create a **high-value second path** for A02/A07 using pretrained recurrence instead of training an entire looped model locally. Engine support and loop switching must be verified; do not assume full 256K context or model fine-tuning works on 12 GB.

## Best-first recommendation

1. Run `experiments/kv_budget.py` to sanity-check VRAM scenarios.
2. Implement B02 page trace format and synthetic unit tests **before** model tracing.
3. Benchmark an LRU and a Quest-style reactive selector, then the predictor, then oracle.
4. If B02 is uninteresting, pivot to C03 rare-fact retention or A04 causal latent communication timing.
