# Local environment observations

For **remote PC** inventory and verified NVIDIA GPU data, see [pc-hardware.md](pc-hardware.md). The shell observations below describe the originating Minimal MCP tool host, **not** the fleet GPU worker.

Observed using Minimal MCP on 2026-11-03:

- Host shell: Windows.
- `python` is callable and unit tests pass.
- `git` installed: version 2.53.0.windows.4.
- `torch` **not installed** in this Python environment.
- `nvidia-smi` was **not executable by name** in this tool environment (CreateProcess Win32 2). This is **not proof of missing GPU hardware**; it may be absent from PATH or installed elsewhere.
- No GPU benchmark or model training executed.

## Before running studies

1. Confirm actual GPU and VRAM using local Windows Device Manager, the NVIDIA app, or full path to nvidia-smi.
2. Confirm available RAM, SSD capacity/speed, CUDA driver, Windows vs WSL2, supported PyTorch wheels.
3. Install dependencies in a project virtual environment; do not modify the Minimal MCP service's own Python environment.
4. Run a smoke load of the chosen checkpoint and record real peak VRAM and decode latency.
5. Treat the resource estimates in `4070ti-feasibility.md` as provisional until the profile is recorded.

## Commands to capture

```powershell
Get-Command nvidia-smi -ErrorAction SilentlyContinue
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
python --version
git --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Note: `Win32_VideoController.AdapterRAM` can underreport modern large VRAM, so verify with NVIDIA tooling.
