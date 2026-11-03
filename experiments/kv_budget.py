"""Conventional transformer KV-cache budget calculator (not a GPU profiler).

Bytes/token = K and V * layers * kv_heads * head_dim * dtype_bytes.
Does NOT include model weights, CUDA runtime, temporary activations, allocator,
quantization metadata, attention kernel workspace or model-specific KV layouts.
"""
from __future__ import annotations

import argparse


def kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, dtype_bytes: int) -> int:
    vals = dict(layers=layers, kv_heads=kv_heads, head_dim=head_dim,
                tokens=tokens, dtype_bytes=dtype_bytes)
    if any(not isinstance(v, int) or v <= 0 for v in vals.values()):
        raise ValueError("all KV shape and dtype parameters must be positive integers")
    return 2 * layers * kv_heads * head_dim * tokens * dtype_bytes


def fmt_gib(n: int) -> str:
    return f"{n / (1024 ** 3):,.3f} GiB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--dtype-bytes", type=int, default=2)
    p.add_argument("--vram-gib", type=float, default=12.0,
                   help="Total board VRAM (not free usable VRAM), default: 12")
    a = p.parse_args()
    if a.vram_gib <= 0:
        p.error("vram-gib must be positive")
    try:
        b = kv_bytes(a.layers, a.kv_heads, a.head_dim, a.tokens, a.dtype_bytes)
        per = kv_bytes(a.layers, a.kv_heads, a.head_dim, 1, a.dtype_bytes)
    except ValueError as ex:
        p.error(str(ex))
    print(f"KV bytes/token: {per:,} B ({per / 1024:.2f} KiB)")
    print(f"KV bytes total: {b:,} B ({fmt_gib(b)})")
    print(f"Board VRAM: {a.vram_gib:.2f} GiB; KV alone = {100*b/(a.vram_gib*1024**3):.1f}%")
    if b >= a.vram_gib * 1024**3:
        print("WARNING: KV alone meets/exceeds nominal board capacity.")
    print("CAUTION: excludes model weights, CUDA allocations, activations, and runtime overhead.")


if __name__ == "__main__":
    main()
