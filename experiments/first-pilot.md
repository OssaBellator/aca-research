# First local pilot — B02 predictive KV-page prefetch

**Status:** planned, no benchmark executed. This is intentionally **not** the full ACA architecture.

## Question

Given past attention/query signals, can a low-cost predictor prefetch pages earlier than a reactive query-aware method, sufficiently to offset routing and transfer overhead on a hypothetical or measured single-GPU memory hierarchy?

**Key novel point to test:** predictive advantage at fixed exact-page budget and real memory transfer cost, *not* merely better hit rate on a synthetic trace.

## Two-phase design

### Phase 1 — trace-driven CPU study

Generate episodes with (i) stable topics, (ii) abrupt topic switches, (iii) delayed rare facts, (iv) alternating-topic thrashing, (v) noisy unrelated items. Trace schema:

```json
{"session":"s001","step":0,"query_features":[0.2,-0.1],"available_pages":[1,2,3],"needed_pages":[1],"page_bytes":65536,"next_step_ms":0.7}
```

Real attention traces, if collected later, should be gathered with **consent-free synthetic tasks** and suitable privacy protections. `needed_pages` on a real transformer must be operationalised (e.g. pages whose removal changes attention output/task quality, not merely the largest raw attention weight).

Compare at equal capacity/preview horizon:

1. LRU cache.
2. Reactive current-query page ranking (Quest-style; exact implementation details documented).
3. Simple Markov/history predictor.
4. Small learned predictor (logistic/MLP; training and inference costs counted).
5. Oracle with perfect future access (upper bound, not deployable).

Evaluate at page budgets 5%, 10%, 25% and varying hypothetical PCIe/host-memory transfer costs; **measure actual transfer latencies before reporting hardware benefit**. Simulate prefetch while reasoning is active; disallow access to future labels during online inference.

### Phase 2 — small-model test (conditional)

Extract reproducible attention/query traces from a small model on short-to-medium context tasks, validate page relevance with controlled page masking, then replay against calibrated transfer latencies. Where kernel integration is feasible, measure end-to-end token latency on hardware.

Never claim that trace-replay results are kernel-level latency improvements.

## Primary metrics

- Stall time after known processing overlap (ms/1000 requests).
- Bytes prefetched but unused; PCIe bytes/second.
- Page hit@k and weighted missed-important-page rate.
- Router/predictor time and energy if available.
- Effect on task answer quality after sparse page selection (secondary GPU follow-up).
- 95% paired bootstrap intervals by independent episode.

## Minimal criteria

A useful positive result requires a pre-registered operating regime with lower total stall than tuned baselines at equivalent page budget, with a reproducible second trace type and an overhead-inclusive estimate. A useful negative result quantifies when anticipation ceases to pay due to bandwidth, inaccurate forecasts or cache thrashing.

## Confounds to prevent

- Artificially deterministic episode sequences making future prediction trivial.
- Oracle leakage into predictor features.
- Counting page-hit gains while ignoring incorrect answer accuracy.
- Treating synthetic latency as a measurement.
- Overfitting same trace generator across train/test.
- Marking different KV position/prefix histories as identical cache pages.

## Budget

Begin with a CPU-only simulator (hours of researcher time; near-zero GPU need). If promising, cap first GPU instrumented pass at a **locally measured** practical resource limit, provisionally 1–6 single-card GPU hours. This is not a promised duration.

## Next action

Implement/verify the trace generator and 4 baselines on synthetic data, then profile GPU/host transfer latencies. Do not train the learned predictor until the simple baselines are stable.

## Second independent pilot

For the more novel cognitive claim, run **A04/A03** on tiny synthetic recurrent networks. Compare early/mid/late gated exchanges, independent best-of-two voting and serial loops at matched FLOPs, followed by stream-message intervention tests. Its practical risk is significantly higher than B02.
