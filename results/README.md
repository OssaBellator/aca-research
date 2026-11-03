# Results registry

**Status: no ACA experiments executed yet.**

When running a study, create `results/<hypothesis-id>/<date>-<run-id>/` containing:

- `config.json`: model/checkpoint hash, framework, CUDA/driver, OS, GPU, seeds, dataset/split hashes, fixed budgets.
- `metrics.csv`: per-item results (including failures), resource timings, optional energy.
- `summary.md`: setup, baselines, confidence intervals, observed failure cases, interpretation.
- `profiling.txt`: `nvidia-smi`, torch and kernel/library versions; peak GPU/host RAM.
- `decision.md`: continue/revise/stop with justification.

Always distinguish *design*, *simulation*, *benchmark*, *experiment*, and *replication*. Do not label projected or toy results as measured hardware outcomes.
