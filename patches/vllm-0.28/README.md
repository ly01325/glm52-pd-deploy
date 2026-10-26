# Dual PCIe patch for Dynamo 1.5.1 / vLLM 0.28.0

Source: the original image builder's patch directory. Targets vLLM 0.28.0 only.

| File | Purpose |
|---|---|
| `apply_all.py` | Idempotent one-step patcher (use this). `python3 apply_all.py --root <site-packages>/vllm --backup-dir <dir> [--dry-run] [--check]` (`--check` lists every anchor that does not match exactly once; used by `scripts/check_patch_compat.py`) |
| `TARGET.json` | Machine-readable target: vLLM version/series, python, vllm root, pip pins. `scripts/ngc_info.py` / `auto_build.py` discover patch sets from this file |
| `final/` | Patched vLLM files + new `b12x_pcie_dma.py`, `flashinfer_pcie_ipc_all_reduce.py` |
| `original/` | Unmodified vLLM 0.28.0 files (reference) |
| `apply_patch.py`, `apply_patch_remaining.py` | Builder's original scripts (run together, in order; not re-runnable) |
| `prefill.env`, `decode.env` | Role env vars |

Verified: `apply_all.py` applied to `original/` reproduces `final/` byte for byte; a second run is a no-op.
Not verified: behaviour inside a real container (no docker/GPU on the authoring machine).
