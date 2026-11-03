# PC fleet device — measured hardware

**Measured:** 2026-11-03 13:51:48 UTC.  
**Source:** Minimal MCP fleet agent, target display name `PC`; read-only `scripts/release/release-readiness.mjs` executed in an isolated workspace snapshot. Remote job exit code **0**, stdout JSON verified.  
**Project link:** The ACA snapshot was transferred to the device for this task. This is **not** a permanent `aca-research` checkout or project registration on PC. The probe used the already registered `minimal-mcp` dependency runtime to execute a dependency-free Node script.

| Property | Observed value |
|---|---|
| GPU | **NVIDIA GeForce RTX 4070 Ti** |
| GPU memory (nvidia-smi) | **12,282 MiB** |
| NVIDIA driver | **610.88** |
| CPU | 11th Gen Intel Core i5-11600K @ 3.90 GHz |
| Logical CPUs | 12 |
| System RAM | 31.81 GiB |
| Free RAM at probe | 16.41 GiB (time-varying) |
| OS | Windows, x64, release 10.0.26300 |
| Node.js | v22.22.3 |
| Python | 3.11.15 |
| PyTorch in probed Python environment | **Not installed** |

Windows WMI reported a lower `AdapterRAM` value, a known limitation for modern GPUs. Use the NVIDIA CLI result for VRAM capacity.

## Consequences for ACA

- **B02 / C03 / B05 / B11**: CPU-first studies remain feasible. Memory/IO simulation can use host RAM; report free RAM at run time.
- **A04 / A02 / A03**: small recurrent-model experiments may be feasible after creating a reproducible virtual environment and profiling. No model training result is implied.
- **Real KV offload**: the GPU has approximately 12 GiB VRAM; available memory after model weights, CUDA/runtime and activations may be substantially smaller. Profile model-specific KV shape.
- **Avoid heavy concurrent workloads** on PC until free RAM/VRAM and thermal limits are checked.

## Remaining prerequisites

1. Explicitly register `aca-research` as a persistent compute project on PC **if** remote training or ongoing tasks are needed. The present readiness run used an isolated ACA snapshot, not a registered checkout.
2. Create a project-local Python virtual environment with a supported PyTorch CUDA wheel if GPU experiments are authorised.
3. Verify `torch.cuda.is_available()` and run a tiny bounded GPU-memory and throughput smoke test.
4. Record CUDA build compatibility, available SSD space and sustained PCIe transfer performance before offload claims.

## Provenance

Remote Minimal MCP fleet job completed successfully. Internal runtime job identifiers are intentionally omitted from the public repository.  
Read-only inventory script: `scripts/release/release-readiness.mjs`.

No GPU training, CUDA microbenchmark, model loading, installations or permanent remote workspace changes were performed.
