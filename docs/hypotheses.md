# Independent ACA hypotheses and required tests

**Status:** designed, NOT executed. Each item is independently falsifiable and can be dropped without abandoning other tracks. Track A = cognition; B = memory; C = consolidation; D = integration. Priorities reflect feasibility on a **12 GB RTX 4070 Ti** and likely value of a clean negative or positive result, not guaranteed gains.

## Common protocol

Pre-register model/checkpoint hashes, datasets, prompt generation, seeds, training steps, budgets, and acceptance criteria; test on held-out splits and matched FLOPs or actual hardware cost. Every test must count training/search costs separately. See [evaluation-protocol.md](evaluation-protocol.md). Numeric hour bands below are **planning ranges, not measured runtimes**. Reports should record limits and counterexamples, including tiny models that fail at baseline tasks.

## Track A — computation and communication

### A01 — Fixed-depth latent recurrence

**Hypothesis:** Repeating a shared latent block improves compositional reasoning under matched FLOPs.

**Required test:** Toy graph traversal, parity and nested Boolean tasks across unseen depth; vary 1/2/4/8 recurrent steps.

**Controls:** Nonrecurrent equal-FLOP network and explicit intermediate-token reasoning.

**Measures:** OOD exact match vs depth, FLOPs, runtime, 3+ seeds.

**Disconfirming result:** No OOD gain against strong equal-cost control.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** 2-8 GPU-h proof-of-concept; longer full study (unprofiled).

### A02 — Predictive cycle budgets

**Hypothesis:** An input-conditioned router selects recurrent depths more efficiently than a fixed budget.

**Required test:** Generate tasks with varied known difficulty; train router on held-out best-budget labels; evaluate at fixed mean FLOPs.

**Controls:** Fixed mean budget; random allocation; oracle upper bound.

**Measures:** Accuracy vs actual FLOPs/latency and budget-prediction regret.

**Disconfirming result:** No Pareto improvement or oracle gap closed negligibly.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU/toy or <=8 GPU-h (unprofiled).

### A03 — Intermediate stream communication

**Hypothesis:** Exchanging partial latent states improves results beyond independent streams + voting.

**Required test:** Train 2-stream network with gated cross-attention; test with communication ablated, shuffled and delayed at inference.

**Controls:** Two independent streams + vote; late fusion; one serial loop at matched FLOPs.

**Measures:** Accuracy, failure correlation, cross-message causal effects, FLOPs, latency.

**Disconfirming result:** Gain disappears vs voting/serial loop or after message scrambling.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Toy <=8 GPU-h; language-scale expensive (unprofiled).

### A04 — Communication timing

**Hypothesis:** Early or intermediate exchange is more useful than final-only fusion for dependency-heavy tasks.

**Required test:** Insert information that stream 2 needs from stream 1 at variable step; sweep exchange cycles.

**Controls:** Always communicate; only final fusion; no exchange.

**Measures:** Accuracy on staged dependency tasks; message bytes/latency.

**Disconfirming result:** No extra benefit from intermediate exchange.

**4070 Ti feasibility:** High. **Initial resource expectation:** Toy CPU/GPU (unprofiled).

### A05 — Complementary latent specialisation

**Hypothesis:** Streams can learn nonredundant solution methods without human-assigned roles.

**Required test:** Evaluate stream-specific error overlap and task transfer with and without diversity regulariser.

**Controls:** Identical initialisation; unspecialised streams; independent samples.

**Measures:** Error correlation, intervention importance, cross-task performance.

**Disconfirming result:** Representations differ but accuracy does not improve.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Small trained model (unprofiled).

### A06 — Router load balancing

**Hypothesis:** Occupancy regularisation prevents collapse without forcing useless streams.

**Required test:** Sweep router entropy/load regularisation on varied tasks; inspect assignments and downstream utility.

**Controls:** No regularisation; uniform mandatory routing.

**Measures:** Active streams, useful contribution, cost-adjusted accuracy.

**Disconfirming result:** Balanced allocation with no generalisation or cost benefit.

**4070 Ti feasibility:** High. **Initial resource expectation:** Toy model (unprofiled).

### A07 — Requests as underallocation labels

**Hypothesis:** Only *successful* extra-cycle requests improve future allocation prediction.

**Required test:** Allow requests during teacher rollouts; label when marginal answer improvement occurs; distil initial router.

**Controls:** Penalise all requests; no request signal; full oracle sweep.

**Measures:** Held-out regret vs compute cost; false request rate.

**Disconfirming result:** Initial budget prediction does not improve.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU-first (unprofiled).

### A08 — Budget-conditioned router

**Hypothesis:** One model can adapt to compute ceilings without separate retraining.

**Required test:** Train with multiple budget tokens and vary ceilings at test time.

**Controls:** Dedicated per-budget routers; fixed policy.

**Measures:** Pareto dominance and calibration at unseen budgets.

**Disconfirming result:** Conditioning adds no benefit or collapses low budget.

**4070 Ti feasibility:** High. **Initial resource expectation:** Small toy (unprofiled).

### A09 — Adaptive stopping

**Hypothesis:** A learned stopping rule avoids wasted cycles without truncating hard tasks.

**Required test:** Label marginal benefit of extra loops on task family; train stop head and calibrate abstention.

**Controls:** Fixed steps; confidence threshold; oracle stop.

**Measures:** Accuracy and expected cycles; false stops on hard cases.

**Disconfirming result:** False stops eliminate efficiency advantage.

**4070 Ti feasibility:** High. **Initial resource expectation:** Toy/GPU (unprofiled).

### A10 — Controller overhead

**Hypothesis:** Net benefit persists after router and synchronisation overhead.

**Required test:** Profile actual GPU kernels for 1/2/4 streams and overlap scheduling, including routing.

**Controls:** Naive serial loops; batched replicas with voting.

**Measures:** Wall-clock, GPU occupancy, peak VRAM and Joules.

**Disconfirming result:** Reduced theoretical FLOPs but worsened latency/energy.

**4070 Ti feasibility:** High. **Initial resource expectation:** Microbenchmark (unprofiled).

### A11 — Long-horizon state integrity

**Hypothesis:** Recurrent stream state retains long causal dependencies without external token traces.

**Required test:** Staged delayed-variable tasks with context separation; intervene by zeroing hidden state.

**Controls:** External scratchpad, chain-of-thought, recurrent no-memory.

**Measures:** Exact match by delay, state perturbation effect.

**Disconfirming result:** No benefit beyond a summary/scratchpad.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Toy <=8 GPU-h (unprofiled).

### A12 — Train-time branching versus inference branching

**Hypothesis:** Distilling exploratory branching into initial allocations retains quality with lower scheduler overhead.

**Required test:** Train with branch requests and teacher search; compare deploy static allocation vs adaptive branch.

**Controls:** Adaptive branch at test; fixed router; supervised labels only.

**Measures:** End-to-end latency, FLOPs, accuracy.

**Disconfirming result:** Static allocation loses most gain.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Small model (unprofiled).

## Track B — memory routing, cache and storage

### B01 — Sparse memory index with pointers

**Hypothesis:** A small semantic/index representation can trigger retrieval of otherwise inaccessible exact facts.

**Required test:** Generate long episodic records with rare delayed relevance; retain index plus exact source store.

**Controls:** Recency, embedding search, keyword index, full-context oracle.

**Measures:** Recall@k, answer accuracy, index bytes, rare-fact miss.

**Disconfirming result:** Miss rate increases disproportionately for rare facts.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU-first (unprofiled).

### B02 — Predictive KV-page prefetch

**Hypothesis:** Past queries/activations predict future page demand early enough to hide transfer latency.

**Required test:** Collect attention-page traces; train low-capacity predictor to prefetch at horizon 1/2/4; simulate measured PCIe latencies.

**Controls:** LRU, recency, reactive query-aware, perfect-foresight oracle.

**Measures:** Stall ms, wasted bytes, hit@k, task-loss proxy, simulation sensitivity.

**Disconfirming result:** No improvement after bandwidth and prediction overhead.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** Best first candidate; CPU trace study (unprofiled).

### B03 — Query-aware KV page selection

**Hypothesis:** Quest-style page statistics beat recency eviction for delayed-relevance tasks on local hardware.

**Required test:** Reimplement page importance from key extrema/query; test fixed page budgets and delayed needles.

**Controls:** LRU, random, last-k, full exact attention.

**Measures:** Top-k target coverage, task accuracy, page-scan cost.

**Disconfirming result:** Equivalent or worse than simple baseline.

**4070 Ti feasibility:** High. **Initial resource expectation:** 1-4 GPU-h optional (unprofiled).

### B04 — Bounded index plus exact archive

**Hypothesis:** Full source storage with compact index reduces active KV without irreversible detail loss.

**Required test:** Small model executes long Q&A with external exact retrieval and fixed hot cache; permute rare-fact placements.

**Controls:** Summary-only memory; retriever without hot cache; full-context if fits.

**Measures:** Exact-fact recall, peak VRAM, I/O latency.

**Disconfirming result:** Archive cannot be found in time or retrieval cost dominates.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU/GPU (unprofiled).

### B05 — Cache-version metadata

**Hypothesis:** Versions and state fingerprints prevent invalid reuse of cached representations.

**Required test:** Construct cache blocks under modified weights, positions and prefixes; attempt use with/without validation.

**Controls:** Unchecked cache reuse.

**Measures:** Mismatch detection; false acceptance/rejection; end-to-end correctness.

**Disconfirming result:** Validator permits incompatible reuse.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** CPU-only test (unprofiled).

### B06 — KV offload threshold

**Hypothesis:** Offload yields lower peak VRAM when KV exceeds working budget, without excessive latency.

**Required test:** Sweep context and concurrency; memory-gated CPU/offload; distinguish prefill/decode and host RAM.

**Controls:** GPU-only, recomputation, simpler paging.

**Measures:** Peak VRAM, p50/p95 latency, tokens/s, bytes transferred.

**Disconfirming result:** Slowdown outweighs added feasible capacity.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** 4070 Ti, needs real pressure (unprofiled).

### B07 — Retrieve or recompute decision

**Hypothesis:** Hardware-aware choice between DMA KV and recomputation outperforms either fixed policy.

**Required test:** Measure transfer and recompute times over block sizes; train simple decision tree and verify held-out workloads.

**Controls:** Always-fetch, always-recompute, hand-coded threshold.

**Measures:** Latency / energy at same correctness.

**Disconfirming result:** A fixed threshold matches learned model.

**4070 Ti feasibility:** High. **Initial resource expectation:** Microbenchmark (unprofiled).

### B08 — Prefetch based on latent signals

**Hypothesis:** Intermediate activations add predictive information beyond recent page access.

**Required test:** Extract hidden/query projections prior to attention; compare trace predictor with and without latent features.

**Controls:** Markov past-access predictor; query-only; oracle.

**Measures:** Out-of-distribution recall@k, stall and model overhead.

**Disconfirming result:** No gain over access-history-only features.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Small model needed (unprofiled).

### B09 — Sparse attention fidelity

**Hypothesis:** Approximating with top-k pages maintains task quality within memory budget.

**Required test:** Run selected-page inference at 10/25/50% pages with exact reference where feasible.

**Controls:** Full attention; recency and random selection.

**Measures:** Logit KL, target accuracy, VRAM, decode time.

**Disconfirming result:** Rare fact failures or degradation exceed preset margin.

**4070 Ti feasibility:** High. **Initial resource expectation:** Small context first (unprofiled).

### B10 — Softmax remote partition merge

**Hypothesis:** Distributed partial attention can reproduce dense attention to numerical tolerance.

**Required test:** Generate disjoint K/V partitions and merge outputs via log-sum-exp; compare torch dense attention.

**Controls:** Unnormalised average/naive sum.

**Measures:** Max abs/relative error in fp32/fp16, operation bytes.

**Disconfirming result:** Cannot reproduce expected tolerance across tested shapes.

**4070 Ti feasibility:** High. **Initial resource expectation:** Single-GPU simulation (unprofiled).

### B11 — Cold-memory thrashing

**Hypothesis:** Page-level history or hysteresis prevents oscillation under alternating topics.

**Required test:** Synthetic queries switch between memory clusters; compare eviction/prefetch policies.

**Controls:** Pure LRU, no hysteresis.

**Measures:** Faults/sec, bytes moved, task errors.

**Disconfirming result:** Policy oscillates or incurs higher cost.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** CPU simulator (unprofiled).

### B12 — Memory-constrained scheduling

**Hypothesis:** Moving compute toward memory beats moving KV for repeated remote accesses under favourable bandwidth.

**Required test:** Cost model with measured transfer/KV sizes; optional multi-node simulator.

**Controls:** Always move memory; always recompute.

**Measures:** Cost / delay by topology; sensitivity plots.

**Disconfirming result:** No feasible operating regime for single-node budget.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Simulation only; real multi-GPU out of scope (unprofiled).

## Track C — consolidation and continual learning

### C01 — Sleep-trigger policy

**Hypothesis:** Consolidating when retrieval interference rises beats fixed-period consolidation.

**Required test:** Sequential-memory simulation, schedule compaction by measured misses/pressure.

**Controls:** Fixed interval, never, always.

**Measures:** Future recall, compaction time, storage footprint.

**Disconfirming result:** Trigger adds overhead without reducing future error.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU-first (unprofiled).

### C02 — Consolidation vs summarisation

**Hypothesis:** Reorganised episodic index improves delayed recall more than summary-only memory.

**Required test:** Sequential episodes with rare changing facts; compare exact archive + indexed consolidation to summaries.

**Controls:** Raw retrieval, fixed summaries, full-history oracle.

**Measures:** Answer accuracy by age/frequency, retention bytes.

**Disconfirming result:** No gain over exact retrieval or high rare-fact loss.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** CPU-first (unprofiled).

### C03 — Rare-fact retention

**Hypothesis:** Novelty and uncertainty plus recurrence protect long-tail information better than frequency-only pruning.

**Required test:** Plant sparse critical facts and repeated misleading facts, impose fixed storage budget.

**Controls:** LRU, LFU, random, uniform reservoir.

**Measures:** Rare-critical recall, total storage, false-belief rate.

**Disconfirming result:** Novelty policy worse than reservoir.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** CPU-first (unprofiled).

### C04 — Rehearsal reduces forgetting

**Hypothesis:** Small diverse replay improves retention after sequential updates relative to no replay.

**Required test:** Train tiny adapter sequentially on domains A/B/C with protected test splits.

**Controls:** No replay; reservoir replay; EWC/regularised adapters if feasible.

**Measures:** A accuracy after B/C, B/C accuracy, update FLOPs.

**Disconfirming result:** Replay cost outweighs retention benefit.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Few-hour GPU LoRA; memory-limited (unprofiled).

### C05 — Adapter-only consolidation

**Hypothesis:** Freezing backbone and updating small memory modules gives safer retention than full-model updates.

**Required test:** Same training data and budgets, compare adapter vs full-model modifications at toy scale.

**Controls:** No weight updates; full tuning.

**Measures:** Protected regression, new-task gain, rollback correctness.

**Disconfirming result:** No advantage in retention/cost.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Small-model QLoRA optional (unprofiled).

### C06 — Protected checkpoint rollback

**Hypothesis:** Automated independent evaluation prevents acceptance of harmful online updates.

**Required test:** Inject known degradations into candidate adapters; run protected unseen tests and rollback.

**Controls:** Manual acceptance; train-loss-only gate.

**Measures:** Detection recall, false rejects, recovery time.

**Disconfirming result:** Misses known regressions or gate overfits.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU/small-GPU (unprofiled).

### C07 — Consolidation cost accounting

**Hypothesis:** Sleep improves net lifetime quality only when update cost is amortised.

**Required test:** Compare cost of repeated retrieval vs periodic compaction under projected number of future queries.

**Controls:** Never sleep; regular compaction; cache only.

**Measures:** Cumulative task utility / full lifecycle joules, storage.

**Disconfirming result:** Payback horizon exceeds workload.

**4070 Ti feasibility:** Very high. **Initial resource expectation:** CPU simulator (unprofiled).

### C08 — Stable beliefs under synthetic replay

**Hypothesis:** Replay needs provenance to avoid self-reinforcing false records.

**Required test:** Mix verified and model-generated episodes with controlled falsehoods; consolidate with vs without provenance gating.

**Controls:** Ungated replay; external-only replay.

**Measures:** False belief persistence, accurate recall, update FLOPs.

**Disconfirming result:** Reinforcement of synthetic errors increases.

**4070 Ti feasibility:** High. **Initial resource expectation:** CPU-first (unprofiled).

## Track D — integration and scale

### D01 — Joint compute+memory routing

**Hypothesis:** Joint allocation improves the Pareto frontier beyond independent routers.

**Required test:** Only after A/B positive; train joint vs separate controllers on coupled delayed-dependency tasks.

**Controls:** Independent routers; fixed resource split.

**Measures:** Accuracy / compute/VRAM/latency frontier.

**Disconfirming result:** Interaction terms nonpositive or unstable.

**4070 Ti feasibility:** Low. **Initial resource expectation:** High compute; defer (unprofiled).

### D02 — Resource competition value

**Hypothesis:** Explicit shared budget yields useful trade-offs among streams, hot KV, and consolidation.

**Required test:** Compare fixed partitions and global allocation on tasks varying in memory versus reasoning demands.

**Controls:** Independent fixed budgets and oracle allocations.

**Measures:** Total utility at equal global cost.

**Disconfirming result:** No additional gain after overhead.

**4070 Ti feasibility:** Medium. **Initial resource expectation:** Trace simulator only initially (unprofiled).

### D03 — Integration fragility

**Hypothesis:** Full system remains robust to changing context distributions and simultaneous load.

**Required test:** Shift prompt topics, sequence length and traffic; stress controllers after held-out training.

**Controls:** Best simple production architecture.

**Measures:** p95/99 latency, accuracy regressions, OOM rate.

**Disconfirming result:** Catastrophic tail failures or instability.

**4070 Ti feasibility:** Low. **Initial resource expectation:** Defer (unprofiled).

### D04 — Scale-transfer evidence

**Hypothesis:** Effects seen in toy or 0.5B models remain directional at a larger feasible checkpoint.

**Required test:** Repeat top result at two sizes, matched relative resources and data.

**Controls:** Single size extrapolation.

**Measures:** Effect sizes/CI versus size and data.

**Disconfirming result:** Gain changes sign or vanishes.

**4070 Ti feasibility:** Low. **Initial resource expectation:** Cloud access likely (unprofiled).

## Independent publication / negative results

A hypothesis can yield a useful local research note if it establishes a reproducible positive result, a well-controlled null result or a precise operating boundary. Report unsuccessful reproductions and known alternatives. Avoid bundling independent claims into one averaged ACA score.
