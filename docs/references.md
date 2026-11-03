# Research references and provenance

Checked on 2026-11-03. These papers establish **adjacent techniques**, not ACA as a whole. Re-check code availability, licenses and version requirements before running.

## Verified primary research

1. **COCONUT — Training Large Language Models to Reason in a Continuous Latent Space** (2024/25): https://arxiv.org/abs/2412.06769 — recurrent hidden-state reasoning. Does *not* establish multiple communicating streams.
2. **MoDr — Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning**, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a81180649011d1690e05c68bfeaa57c-Abstract-Conference.html — learned routing of multiple branches; strong competitor.
3. **Depth-Recurrent Attention Mixtures** (2026 preprint): https://arxiv.org/abs/2601.21582 — compares recurrent attention arrangements and controls for memory/parameters.
4. **Quest — Query-Aware Sparsity for Efficient Long-Context LLM Inference**, ICML 2024: https://proceedings.mlr.press/v235/tang24l.html — query-aware sparse KV page selection. Note that the project page and index snippets can transpose attention-only and end-to-end speedup numbers; quote metrics from the reviewed paper after checking.
5. **Titans: Learning to Memorize at Test Time**, Google Research 2025: https://research.google/pubs/titans-learning-to-memorize-at-test-time/ — learned persistent neural memory.
6. **Decode Context Parallelism**, vLLM technical write-up, 2026-08-07: https://vllm.ai/blog/2026-08-07-decode-context-parallelism — distributed exact-sequence attention at a *multi-GPU* serving scale; not evidence of benefit on a single 4070 Ti.
7. **RULER**, NVIDIA research benchmark: https://github.com/NVIDIA/RULER — synthetic context/recall and multi-hop tasks; not by itself realistic long-horizon utility.
8. **BABILong**: https://arxiv.org/abs/2406.10149 — distributed-fact reasoning in long context.
9. **RTX 4070 family**, NVIDIA official specifications: https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4070-family/ — 4070 Ti: 12 GB GDDR6X; 4070 Ti SUPER: 16 GB.

## New October 2026 candidate: Dory 1.0 4B

- **NVIDIA Dory / Elastic Latent Recurrence**: https://arxiv.org/abs/2610.12582 and https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 — pretrained 4B recurrent transformer with variable loop count and last-loop KV cache. Vendor-reported benchmark results require independent reproduction; see [dory-evaluation.md](dory-evaluation.md) for ACA fit and caveats.
- **NVIDIA Q5_K_M checkpoint**: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF — 3,673,288,864-byte quantised model for a potential 12 GB local inference experiment. The NVIDIA-documented llama.cpp support path depends on a Dory-enabled revision, not necessarily a standard release.

## Supporting literature requiring experiment-specific verification

- **Mixture-of-Recursions** (2025) and **LOTUS** (2026): useful comparisons of adaptive computation and parallel recurrent latent reasoning; read code/configuration carefully before claiming apples-to-apples FLOPs.
- **LongMemEval**: https://arxiv.org/abs/2410.10813 — long-term conversational memory evaluation; model support and data processing details should be confirmed.
- **Language Models Need Sleep** (2026 preprint): https://arxiv.org/abs/2606.03979 — a relevant consolidation analogy, not proof that autonomous overnight weight updates improve arbitrary tasks.

## Evidence categories

- **Established external**: actual published result; cite exact setup and limitations.
- **Proposed here**: hypothesis, protocol or engineering concept; no experiment conducted.
- **Estimate**: memory formula or time range; list assumptions and verify by profiling.
- **Open**: transfer to new models, causal inter-stream benefit, continual-learning stability.

## Avoid

Do not report prior conversation's detailed dates, accuracy gains or throughput figures as newly verified without reading the primary experimental methods. Do not conflate exact cache storage with exact dense attention when sparse selection is used.
