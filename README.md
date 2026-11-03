# ACA — Adaptive Cognitive Architecture

**A falsifiable research programme for resource-constrained latent reasoning, predictive memory, and continual learning.**  
Status: **hypotheses and experimental designs only** — no ACA performance results yet. Updated: 2026-11-03.

## Objective

Test whether a neural model can improve *quality per total computation* by learning (a) when to run multiple recurrent latent streams, (b) when they should share partial information, (c) which exact historical representations should be resident or fetched, and (d) when to consolidate or forget knowledge.

This repository **separates four tracks** rather than treating ACA as one all-or-nothing project:

| Track | Focus | Can it start independently? |
|---|---|---|
| A — Cognition | Recurrence, parallel latent states, learned compute/communication routing | Yes |
| B — Memory | KV-page selection, predictive prefetch, offload and cache integrity | Yes |
| C — Consolidation | Bounded episodic memory, replay, adapter learning, checkpointing | Yes, using ordinary models |
| D — Integration | Joint compute–memory–consolidation policies | **Only after** component evidence |

## Start here (single RTX 4070 Ti)

The **RTX 4070 Ti has 12 GB VRAM** (the 4070 Ti **SUPER has 16 GB**). A laptop or differently named GPU must be measured, not assumed. See [docs/4070ti-feasibility.md](docs/4070ti-feasibility.md). The first recommended study is **B02: predictive versus reactive KV-page prefetching**, as a trace-driven simulation, followed by an optional small-model test. This requires **no expensive model training** and may produce a reproducible result even if predictive routing fails.

A more scientifically central but riskier follow-up is **A03: does intermediate latent stream communication beat voting at matched FLOPs?** Begin with a toy recurrent network and synthetic algorithmic tasks before training a language model.

**Do not infer H100-hours by scaling FLOP counts to a consumer GPU.** Profile the exact kernels, model, batch, Windows/WSL configuration and memory pressure before making runtime estimates.

## Project files

- [docs/architecture.md](docs/architecture.md) — proposed system, interfaces, invariants, dependencies.
- [docs/hypotheses.md](docs/hypotheses.md) — **36 independently falsifiable hypotheses**, each with controls, metrics and failure conditions.
- [docs/4070ti-feasibility.md](docs/4070ti-feasibility.md) — ranked local experiments, VRAM calculation, time/resource risk.
- [docs/local-environment.md](docs/local-environment.md) — local tool-host limitations.
- [docs/pc-hardware.md](docs/pc-hardware.md) — **measured remote PC GPU and software inventory**.
- [docs/evaluation-protocol.md](docs/evaluation-protocol.md) — preregistration, statistics, leakage tests, cost accounting.
- [docs/references.md](docs/references.md) — verified literature and gaps.
- [docs/dory-evaluation.md](docs/dory-evaluation.md) — NVIDIA Dory 4B as a practical ACA recurrent-router testbed.
- [experiments/first-pilot.md](experiments/first-pilot.md) — concrete low-cost B02 pilot, baselines and go/no-go.
- [experiments/hypothesis-index.csv](experiments/hypothesis-index.csv) — machine-readable hypothesis inventory.
- [experiments/kv_budget.py](experiments/kv_budget.py) — dependency-free KV capacity calculator.
- [results/README.md](results/README.md) — reporting template and status rules.

## Guiding requirements

1. **Causal effect**, not merely attractive latent representations: interventions must change outcomes.
2. **Strong baselines**: fixed recurrence; serial equivalent; independent parallel streams + voting; MoDr where feasible; LRU, Quest-style query-aware, and oracle memory scheduling.
3. **Matched cost**: report training, hyperparameter search, inference FLOPs, end-to-end latency, VRAM peak, CPU RAM, SSD bytes, transfer volume and energy where available.
4. **Bounded resources**: the router's inference cost and offline training/maintenance must enter the ledger.
5. **Cache correctness**: 'exact KV' is exact only relative to compatible weights, token positions, masks and computation history. Sparse selection itself is approximate.
6. **Independent results**: a failed branch does not invalidate unrelated tracks. Negative results should be released with artifacts.

## Minimal hardware sanity check

Run on the target machine:

```powershell
nvidia-smi
python experiments/kv_budget.py --layers 24 --kv-heads 8 --head-dim 64 --tokens 131072 --dtype-bytes 2
```

The script calculates **only KV tensor storage**, not model weights, logits, activations, allocator overhead or CUDA runtime. No actual hardware or training benchmark has been run for this repository.

## What constitutes a useful result?

A useful publication need not show a new frontier model. Clear evidence that a predictive router fails to beat Quest-style selection under plausible bandwidth or that latent communication fails against voting is itself worth documenting if the setup is fair and reproducible.

## Research context

This proposal was developed from a discussion of recurrent latent streams, learned compute allocation, dynamic KV memory, distributed attention and sleep-like consolidation. It incorporates feedback to keep the serving and ML experiments independent. Links to original work and precise scope are in [docs/references.md](docs/references.md).

**License and reuse:** MIT License — see [LICENSE](LICENSE). Model checkpoints retain their own licences; no third-party model weights are included.
