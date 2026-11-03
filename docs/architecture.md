# ACA architecture — working design

## Boundary of claims

ACA is a hypothesis, **not** an implemented or validated system. Research on recurrence, routing, sparse attention, distributed KV and test-time memory supports individual components, not a jointly trained organism-like cognition. Stronger causal evidence is required for claims about 'thought streams'.

## Conceptual data flow

```text
                         BUDGET (FLOPs / VRAM / latency / storage)
                                        |
                         +--------------v-------------+
input, context ----------> compute router / controller|<----- feedback
                         +----+---------------+-------+
                              |               |
                      recurrent states        | memory predictions
                        h1 h2 ... hK          v
                         \  |  /      +---------------------------+
                       gated workspace | memory router / directory |
                         /     \        +--+------------+-----------+
                  answer head  validator  | sparse index| archive
                                         | hot KV <---->| warm/cold exact
                                         | local attention / remote attention
                                         +---------------------------+

offline: logs -> consolidation / rehearsal -> gated candidate checkpoint
                                   -> independent held-out tests -> keep/rollback
```

## Module contracts

**Cognition core:** shared weights θ, per-stream recurrent state `h_i^t`, versioned context ID, stop/budget. Intermediate exchange only through explicit gated cross-attention or workspace. No assumption streams automatically specialise.

**Compute router:** consumes input/task embeddings, stream states, uncertainty proxies, active budgets. Emits stream count, per-stream cycle quotas, communication gates, stop actions. During training, extra-compute requests can label underallocation **only when the extra cycles causally improve task quality**.

**Memory directory:** metadata summaries / query-aware key statistics / provenance pointers; not a substitute for exact content. Maintain stable block IDs, model/tokeniser/version, prefix/state hash, positional configuration, causal mask, dtype and ownership. Do not combine independently processed KV histories as though they formed one dense causal context.

**Storage fabric:** local KV pages, CPU RAM/SSD spill, optional remote attention. Exact bytes may be moved losslessly; querying a subset is still approximate attention. For model changes, invalidate or recompute incompatible caches.

**Consolidation:** separates (1) reversible eviction from hot cache, (2) durable source records, (3) policy-governed deletion, (4) optional parameter/adaptor updates. Consolidation must not erase user-required data-retention/deletion boundaries.

**Evaluator:** immutable held-out tasks, causal interventions, reference checkpoint, independent acceptance criteria and controlled rollback; rollback cannot undo earlier external-world actions.

## Hard constraints and cost accounting

`L = L_task + λ_F * FLOPs + λ_M * peak_VRAM + λ_T * bytes_moved + λ_L * latency + λ_S * retained_storage + λ_E * regressions`.

These are **illustrative surrogate terms**, not a scientifically validated choice of coefficients. Real experiments must separately report Pareto fronts and per-metric baselines; scalarised rewards can hide trade-offs.

## Research dependencies

- **Track A** (latent streams): independent of **Track B** (memory) at first.
- **Track C** (consolidation): test against an existing small model with ordinary episodic memory; don't require A or B.
- **Track D** (integration): only after significant independent A and B wins; may be inferior due to controller overhead.

## What the model is *not*

A stored KV block is not an independent agent or a general-purpose model. Offloading to another computer is useful only when memory/bandwidth trade-offs justify it. 'Sleep' is an offline training/scheduling protocol, not evidence of subjective fatigue.

## Exactness and scale

KV bytes per token `= 2 * layers * kv_heads * head_dim * dtype_bytes` for a conventional transformer cache. Real layout may include padding, quantisation scales, sliding windows, MLA, different key/value dimensions or sharing. `experiments/kv_budget.py` illustrates the simple case.

Remote attention can be merged exactly across disjoint key partitions for compatible attention via local output and log-sum-exp statistics, subject to numerical precision. Returning only chosen pages changes the computation. Reconstructing historical KV from text after a weight update is not generally byte-exact.

## Threats to validity

- Extra streams may add only ensembling, not causal exchange.
- Router may learn task-ID shortcuts or suppress legitimate requests.
- Synthetic tasks may not transfer to language reasoning.
- Selective retrieval may miss rare facts even when summary/index looked irrelevant.
- All methods may be slower on a single PCIe GPU due to transfer overhead.
- Strong baseline implementations and kernel quality can dominate reported gains.
- Offline training/repair can overwhelm inference savings.
