# Evaluation protocol and preregistration

## Unit of comparison

The primary result for a given hypothesis is the **paired difference against its strongest credible baseline**, measured on held-out tasks and at a declared resource budget. Report cost and quality as separate axes, not just a combined score.

Each study registers `id, version, hypothesis, primary_metric, model_hash, dataset_hash, train/validation/test split, baselines, budgets, significance plan, disconfirming criterion`.

## Rules

1. **Matched compute**: report FLOPs estimated from actual operator shapes and measured wall time separately. A routed model can be FLOP-efficient yet slower on GPU.
2. **Full cost ledger**: include acquisition of oracle labels, router tuning, train-time searches, retries, CPU work, SSD/DRAM bandwidth, and amortisation assumption for future queries. Separate per-inference cost from fixed training costs.
3. **Minimum baselines**: always test a strong non-learned control and an optimistic oracle. For cognition include independent candidates+voting and serial recurrence. For memory include recency/LRU and query-aware page selection. For consolidation include no updates and reservoir replay where appropriate.
4. **Avoid leakage**: split at source-document/task-generator/temporal episode level, not only by near-duplicate question IDs. Test on unseen problem lengths and difficulty distributions.
5. **Statistics**: use paired bootstrap 95% confidence intervals over examples or episodes; for trained models, report >=3 seeds where tractable (more if noisy). Show per-seed scatter, not only pooled intervals. Adjust multiple comparisons or clearly label exploratory tests. Don't infer statistical independence from related episodes.
6. **Causal interventions**: remove or scramble stream communication; randomise the sparse index; withhold prefetch; invalidate an exact KV block; disable consolidation. Positive correlation is insufficient.
7. **Negative cases**: include rare future-relevance facts, contradictions, cache thrashing, misleading synthetic data, unseen difficulty, busy hardware and p95/p99 latency.
8. **Model correctness**: when claiming 'exact', compare to local reference with declared tensor precision and tolerances. Equal weights across two nodes do not make independently conditioned KV compatible.
9. **No retroactive targets**: freeze primary success/failure metrics before opening the held-out result.
10. **Security/privacy**: no live user records or secrets in traces; synthetic prompts/data or consented open benchmarks only. Treat sample outputs as untrusted.

## Preregistration template

```yaml
hypothesis: B02
date: 2026-11-03
status: planned
primary_metric: stall_ms_per_1000_requests
secondary_metrics: [hit_rate, wasted_bytes, router_ms, p95_latency]
baselines: [lru, reactive_query_aware, markov, oracle]
data: [held_out_synthetic_trace, optional_real_trace]
cost_budget: {gpu_hours: TBD_after_smoke_profile, max_host_ram_gb: TBD}
success_rule: "lower mean stall and no meaningful p95 regression in ≥1 preregistered workload, with positive full-cost benefit"
failure_rule: "best simple baseline matches or beats predictor after overhead"
seed_policy: "≥3 training seeds for learned predictor; 5 simulated workload seeds"
uncertainty: "paired bootstrap over independent trace sessions, 95% CI"
```

## Measurement ledger

```csv
run_id,hypothesis,model_hash,seed,task_id,accuracy,estimated_flops,gpu_seconds,cpu_seconds,peak_vram_bytes,peak_ram_bytes,transfer_bytes,ssd_bytes,energy_joules,notes
```

Record missing values explicitly; don't silently substitute zero. Include system configuration in every experiment artifact.

## Publication criterion

A reproducible null result with controlled baselines and measured operating regime is useful. A simulation result should be called a simulation, not a hardware throughput measurement. A one-seed success on a toy benchmark is not evidence of general capability scaling.
