# Dory 1.0 4B — ACA candidate model evaluation

**Research note (2026-11-04)** — model inspection / external-source analysis only; **no checkpoint downloaded or GPU run completed**.

## Finding

**High-value candidate for ACA Track A, especially A01/A02/A07/A09.** NVIDIA's looped Dory 1.0 is already trained for flexible recurrent depth, eliminating the need to train the base recurrence from scratch on a 12 GB GPU. It does **not** natively provide independent interacting latent streams (A03/A04), so it is a base to investigate, not a full ACA implementation.

### Primary sources

- Paper: [One Model to Loop Them All: Elastic Latent Recurrence for Language Models](https://arxiv.org/abs/2610.12582), submitted 2026-10-08.
- [Official BF16 model](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16), release 2026-10-16.
- [Official Q5_K_M GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF).
- [NVIDIA's documented Dory-enabled llama.cpp revision](https://github.com/brianchmielNV/llama.cpp/commit/98a74c97ea09ac72f6ceda1559808fc0a2f447ce); NVIDIA states official llama.cpp support was still pending review in PR #30306 at time of model documentation.
- [Dory vLLM PR #60149](https://github.com/vllm-project/vllm/pull/60149) — version/support status must be checked before relying on it.
- License: OpenMDW 1.1 (review exact terms before derivative release or redistribution).

## Architecture relevant to ACA

- 4 billion parameters, nGPT-based with hybrid sliding-window and full attention.
- 8-layer input + 20-layer shared recurrent block + 8-layer output; computational layer traversals at 1/2/4 loops are approximately 36/56/96, **not a precise FLOP ratio**.
- Input is re-injected at each recurrent pass.
- Supports one checkpoint at varying loop counts; NVIDIA suggests 2 loops as default.
- **Per-loop KV**: separate KV per recurrent loop; memory rises with depth.
- **Last-loop KV**: only final recurrent KV held; useful memory-depth trade-off, near but not guaranteed identical accuracy.
- The 256K advertised context is a model capacity, **not a claim it can fit 256K on the 4070 Ti**.

NVIDIA vendor-reported BF16 SciCode: 27.3 / 34.94 / 35.11 at loops 1/2/4; SWE-bench Verified: 50.3 / 59.5 / 61.8. These are **vendor-reported with task-specific harnesses**, not local independent results. Some measures do not improve monotonically with loops; do not assume deeper always better.

## 12 GB device fit

Fleet verified the target `PC`: RTX 4070 Ti with 12,282 MiB VRAM, 31.81 GiB system RAM, driver 610.88, Python 3.11; no PyTorch installed in probed environment. See [pc-hardware.md](pc-hardware.md).

- Q5_K_M GGUF reference file: **3,673,288,864 bytes = ~3.42 GiB** disk/model data before runtime overhead; likely feasible for short-context inference if proper kernel support is available.
- BF16 raw 4B weights: roughly **8 billion bytes / 7.45 GiB**; GPU fit may be tight once cache, compute graph and allocator memory are included. Not a recommended first configuration.
- NVIDIA native **NVFP4** deployment speedups require compatible hardware/kernel support; do not assume Blackwell-native FP4 performance on Ada RTX 4070 Ti.
- Start at **4096-token context, single stream, Q5 GGUF**, one supported engine, explicitly verify full CUDA offload and free VRAM. Repeat 1, 2 and 4 loops at same cache mode.
- Windows CUDA build compatibility of the Dory-enabled llama.cpp revision **has not been independently tested here**. NVIDIA documents Linux CUDA and macOS, while generic Hugging Face Windows app hints may not reflect model-specific loop control. WSL2 may simplify development if already installed; avoid reinstalling system software without need.

## Recommended Dory pilot — A02/A07

**Question:** Can a low-cost prompt-conditioned router allocate 1/2/4 loops more efficiently than choosing a fixed loop count for every item?

1. Install/run Dory GGUF on PC only as a **separate subsequent implementation task**, after capability check. Freeze model and engine revision.
2. Assemble a held-out set of reliably scored problems with easy/medium/hard variants; separate train/validation/test by generator family or source.
3. Evaluate the same questions with loops **1, 2, 4** at fixed seed / deterministic decoding, matched cache mode, context length and output limits. Record answer correctness, input/output token counts, latency, peak VRAM and failure cases. Repeat stochastic evaluations if decoding demands it.
4. Produce a teacher table with `task_id, features_available_before_inference, correct_1, correct_2, correct_4, time_1, time_2, time_4`. **No test labels may enter router training**.
5. Train a tiny classifier on input-only cheap features (e.g. length, shape, difficulty predictors); use validation to select policy parameters. **Do not inspect future model answers or benchmark labels at decision time**.
6. Compare against fixed 1/2/4, random and frequency-matched allocations, an oracle hindsight allocation (nondeployable upper bound), and **independent one-loop samples + voting** / **serial or fixed two-loop baseline** when total compute is comparable.
7. Evaluate **quality at equal realised compute and wall time**; include router overhead, output length changes, cache mode and engine restarts/reconfiguration. If inference engine requires separate server configs or restarts to change loops, a repeated offline sweep can establish *policy potential*, **not** real per-request adaptive serving latency.
8. Apply a preregistered null test: if router cannot beat tuned fixed-budget or ensembling controls on untouched data at matched cost, no claim of learned-depth benefit.

## Second study — memory modes

Within one loop-count setting, benchmark per-loop vs last-loop KV cache at 4K / 8K / 16K **only when model engine and VRAM permit**. Measure retained KV bytes, end-to-end latency and answer/logit disagreement. This can independently establish whether last-loop memory savings justify accuracy trade-offs on 12GB hardware.

## What Dory cannot establish by itself

- **A03/A04**: intermediate communication between *separate persistent recurrent latent streams*. Dory loops are sequential passes of one stream. Implementing multiple parallel branches requires modifying the inference/training computation graph and likely more memory.
- **B02**: exposing meaningful, per-page query / attention traces may require engine instrumentation. GGUF inference APIs alone may not provide hidden states or page-level KV access.
- **C01–C08**: no built-in sleep-like autonomous consolidation. Modifying weights on 4070 Ti would require a new training pipeline, likely adapters and dedicated compatibility evaluation.
- **Scale transfer**: performance at 4B cannot establish transfer to larger models.

## Go/no-go

**Go:** successfully load checkpoint at 4096 context, compare 1/2/4 with identical test conditions, no OOM or silent CPU fallback, and demonstrate that adaptive policy can recover a nontrivial portion of oracle quality–compute gap on unseen questions beyond tuned fixed and sampling baselines.

**Report even if null:** per-example loop efficacy, cases where extra loops hurt, quantisation-dependent accuracy, cache-memory trade-offs and router regret under measured 4070 Ti latency.

**Do not do yet:** train parallel-stream Dory, assume per-token loop switching is already supported, or advertise 256K context as feasible on 12 GB.

## Data and license caution

Do not publish downloaded model weights in ACA. Respect OpenMDW 1.1 and benchmark/dataset licensing. Avoid collecting private user material in prompts, logs or training traces.
