#!/usr/bin/env python3
"""Idempotent, single-step version of the 1.5.1 / vLLM 0.28.0 "dual PCIe" patch.

Combines apply_patch.py + apply_patch_remaining.py + the manual SM120 one-liner:
  * (apply_patch.py stops at its cuda_communicator destroy step - the anchor matches 3 times - and
     apply_patch_remaining.py finishes it; here the destroy step uses the unique anchor directly.)
  * adds `supports_dense_mha_prefill = False` to flashinfer_mla_sparse_sm120.py
  * copies b12x_pcie_dma.py and flashinfer_pcie_ipc_all_reduce.py from ./final/ into device_communicators/
Re-running is safe: already-patched files are skipped. Targets vLLM 0.28.0 ONLY (anchors are exact strings).

usage: python3 apply_all.py [--root /usr/local/lib/python3.12/dist-packages/vllm] [--backup-dir DIR] [--dry-run]
Result is byte-identical to ./final/*.py (verified when this script was written).
"""
import argparse, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/usr/local/lib/python3.12/dist-packages/vllm")
ap.add_argument("--backup-dir", default="/tmp/glm52-vllm-original")
ap.add_argument("--new-files-dir", default=str(Path(__file__).resolve().parent / "final"))
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--check", action="store_true", help="compatibility check: report every anchor that does not match exactly once (never writes, keeps going after a miss)")
args = ap.parse_args()
if args.check: args.dry_run = True

root = Path(args.root)
if not root.is_dir(): sys.exit(f"vllm root not found: {root}")
files = {
    "envs.py": root / "envs.py",
    "cuda_communicator.py": root / "distributed/device_communicators/cuda_communicator.py",
    "parallel_state.py": root / "distributed/parallel_state.py",
    "kernel_warmup.py": root / "model_executor/warmup/kernel_warmup.py",
    "flashinfer_mla_sparse_sm120.py": root / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
}
for n, p in files.items():
    if not p.exists(): sys.exit(f"missing {p} - is this vLLM 0.28.0?")

backup = Path(args.backup_dir); 
if not args.dry_run:
    backup.mkdir(parents=True, exist_ok=True)
    for n, p in files.items():
        if not (backup / n).exists(): shutil.copy2(p, backup / n)

_buf = {}  # edits are staged in memory and written per file only after ALL its edits succeed (no half-patched files)

_ok = {}; _miss = []

def replace_once(path: Path, old: str, new: str) -> None:
    text = _buf.get(path, path.read_text()); count = text.count(old)
    if count != 1:
        if args.check:
            _miss.append((path.name, count, old.strip().splitlines()[0][:90] if old.strip() else old[:90])); return
        raise RuntimeError(f"{path}: expected one match, got {count}: {old[:80]!r} (wrong vLLM version?)")
    _buf[path] = text.replace(old, new, 1); _ok[path.name] = _ok.get(path.name, 0) + 1

def commit(path: Path) -> None:
    if path in _buf and not args.dry_run: path.write_text(_buf[path])

def done(path: Path, marker: str) -> bool:
    return marker in path.read_text()

# 1) envs.py
if done(files["envs.py"], "VLLM_ALLREDUCE_USE_B12X_PCIE_DMA"): print("skip envs.py (already patched)")
else:
    envs = files["envs.py"]
    replace_once(
        envs,
        "    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False\n",
        "    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False\n"
        "    VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC: bool = False\n"
        "    VLLM_ALLREDUCE_USE_B12X_PCIE_DMA: bool = False\n"
        "    VLLM_B12X_DMA_MIN_BYTES: int = 8 * 1024 * 1024\n"
        "    VLLM_B12X_DMA_MAX_BYTES: int = 256 * 1024 * 1024\n"
        "    VLLM_B12X_DMA_FP8: str = \"ring\"\n",
    )
    replace_once(
        envs,
        '    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(\n'
        '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))\n'
        "    ),\n",
        '    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(\n'
        '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))\n'
        "    ),\n"
        '    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": lambda: bool(\n'
        '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"))\n'
        "    ),\n"
        '    "VLLM_ALLREDUCE_USE_B12X_PCIE_DMA": lambda: bool(\n'
        '        int(os.getenv("VLLM_ALLREDUCE_USE_B12X_PCIE_DMA", "0"))\n'
        "    ),\n"
        '    "VLLM_B12X_DMA_MIN_BYTES": lambda: int(\n'
        '        os.getenv("VLLM_B12X_DMA_MIN_BYTES", str(8 * 1024 * 1024))\n'
        "    ),\n"
        '    "VLLM_B12X_DMA_MAX_BYTES": lambda: int(\n'
        '        os.getenv("VLLM_B12X_DMA_MAX_BYTES", str(256 * 1024 * 1024))\n'
        "    ),\n"
        '    "VLLM_B12X_DMA_FP8": lambda: os.getenv("VLLM_B12X_DMA_FP8", "ring"),\n',
    )
    commit(files["envs.py"]); print("patched envs.py")

# 2) cuda_communicator.py
comm = files["cuda_communicator.py"]
if done(comm, "FlashInferPcieIpcAllReduce"): print("skip cuda_communicator.py (already patched)")
else:
    comm = files["cuda_communicator.py"]
    replace_once(comm, '        if "tp" not in unique_name:\n', '        if unique_name.split(":")[0] != "tp":\n')
    replace_once(
        comm,
        "            use_flashinfer_allreduce = False\n            use_aiter_allreduce = False\n",
        "            use_flashinfer_allreduce = False\n"
        "            use_flashinfer_pcie_ipc_allreduce = False\n"
        "            use_b12x_pcie_dma_allreduce = False\n"
        "            use_aiter_allreduce = False\n",
    )
    replace_once(
        comm,
        "            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER\n",
        "            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER\n"
        "            use_flashinfer_pcie_ipc_allreduce = (\n"
        "                envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC\n"
        "                and not envs.VLLM_BATCH_INVARIANT\n"
        "            )\n"
        "            use_b12x_pcie_dma_allreduce = (\n"
        "                envs.VLLM_ALLREDUCE_USE_B12X_PCIE_DMA\n"
        "                and not envs.VLLM_BATCH_INVARIANT\n"
        "            )\n",
    )
    replace_once(
        comm,
        "        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n",
        "        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n"
        "        self.use_flashinfer_pcie_ipc_allreduce = use_flashinfer_pcie_ipc_allreduce\n"
        "        self.use_b12x_pcie_dma_allreduce = use_b12x_pcie_dma_allreduce\n",
    )
    replace_once(
        comm,
        "        # lazy import to avoid documentation build error\n",
        "        # lazy import to avoid documentation build error\n"
        "        from vllm.distributed.device_communicators.b12x_pcie_dma import (\n"
        "            B12xPcieDmaAllReduce,\n"
        "        )\n",
    )
    replace_once(
        comm,
        "        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator\n",
        "        from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import (\n"
        "            FlashInferPcieIpcAllReduce,\n"
        "        )\n"
        "        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator\n",
    )
    replace_once(
        comm,
        "        self.fi_ar_comm: FlashInferAllReduce | None = None\n",
        "        self.fi_ar_comm: FlashInferAllReduce | None = None\n"
        "        self.fi_pcie_ipc_ar_comm: FlashInferPcieIpcAllReduce | None = None\n"
        "        self.b12x_pcie_dma_ar_comm: B12xPcieDmaAllReduce | None = None\n",
    )
    replace_once(
        comm,
        "        if self.use_flashinfer_allreduce and self.world_size > 1:\n",
        "        if (\n"
        "            self.use_b12x_pcie_dma_allreduce\n"
        "            and self.world_size > 1\n"
        "            and self.device_group is not None\n"
        "        ):\n"
        "            self.b12x_pcie_dma_ar_comm = B12xPcieDmaAllReduce(\n"
        "                group=self.device_group, device=self.device\n"
        "            )\n\n"
        "        if self.use_flashinfer_allreduce and self.world_size > 1:\n",
    )
    replace_once(
        comm,
        "        if self.use_aiter_allreduce and self.world_size > 1:\n",
        "        if (\n"
        "            self.use_flashinfer_pcie_ipc_allreduce\n"
        "            and self.world_size > 1\n"
        "            and self.device_group is not None\n"
        "        ):\n"
        "            self.fi_pcie_ipc_ar_comm = FlashInferPcieIpcAllReduce(\n"
        "                group=self.device_group,\n"
        "                tune_group=self.cpu_group,\n"
        "                device=self.device,\n"
        "            )\n\n"
        "        if self.use_aiter_allreduce and self.world_size > 1:\n",
    )
    replace_once(
        comm,
        '        all_potential_ar_backends = [\n            "NCCL_SYMM_MEM",\n',
        '        all_potential_ar_backends = [\n'
        '            "B12X_PCIE_DMA",\n'
        '            "FLASHINFER_PCIE_IPC",\n'
        '            "NCCL_SYMM_MEM",\n',
    )
    replace_once(
        comm,
        "        enabled_ar_backends: list[str] = []\n",
        "        enabled_ar_backends: list[str] = []\n"
        "        if (\n"
        "            self.b12x_pcie_dma_ar_comm is not None\n"
        "            and not self.b12x_pcie_dma_ar_comm.disabled\n"
        "        ):\n"
        '            enabled_ar_backends.append("B12X_PCIE_DMA")\n'
        "        if (\n"
        "            self.fi_pcie_ipc_ar_comm is not None\n"
        "            and not self.fi_pcie_ipc_ar_comm.disabled\n"
        "        ):\n"
        '            enabled_ar_backends.append("FLASHINFER_PCIE_IPC")\n',
    )
    replace_once(
        comm,
        "        fi_ar_comm = self.fi_ar_comm\n",
        "        b12x_pcie_dma_ar_comm = self.b12x_pcie_dma_ar_comm\n"
        "        if b12x_pcie_dma_ar_comm is not None:\n"
        "            out = b12x_pcie_dma_ar_comm.all_reduce(input_)\n"
        "            if out is not None:\n"
        "                return out\n"
        "        fi_pcie_ipc_ar_comm = self.fi_pcie_ipc_ar_comm\n"
        "        if fi_pcie_ipc_ar_comm is not None and fi_pcie_ipc_ar_comm.should_use(input_):\n"
        "            return fi_pcie_ipc_ar_comm.all_reduce(input_)\n"
        "        fi_ar_comm = self.fi_ar_comm\n",
    )
    # destroy(): unique anchor (the all2all_manager anchor used by apply_patch.py matches 3 places)
    old = (
        "        if self.fi_ar_comm is not None:\n"
        "            self.fi_ar_comm.destroy()\n"
        "            self.fi_ar_comm = None\n"
        "        if self.all2all_manager is not None:\n"
    )
    new = (
        "        if self.fi_ar_comm is not None:\n"
        "            self.fi_ar_comm.destroy()\n"
        "            self.fi_ar_comm = None\n"
        "        if self.fi_pcie_ipc_ar_comm is not None:\n"
        "            self.fi_pcie_ipc_ar_comm.destroy()\n"
        "            self.fi_pcie_ipc_ar_comm = None\n"
        "        if self.b12x_pcie_dma_ar_comm is not None:\n"
        "            self.b12x_pcie_dma_ar_comm.close()\n"
        "            self.b12x_pcie_dma_ar_comm = None\n"
        "        if self.all2all_manager is not None:\n"
    )
    replace_once(comm, old, new)
    commit(comm); print("patched cuda_communicator.py")

# 3) parallel_state.py  (same edits as apply_patch_remaining.py, which produced final/)
ps = files["parallel_state.py"]
if done(ps, "maybe_fi_pcie_ipc_context"): print("skip parallel_state.py (already patched)")
else:
    replace_once(ps, "        maybe_ca_context = nullcontext()\n        maybe_aiter_context = nullcontext()\n",
                 "        maybe_ca_context = nullcontext()\n        maybe_fi_pcie_ipc_context = nullcontext()\n        maybe_aiter_context = nullcontext()\n")
    replace_once(ps, "            if ca_comm is not None:\n                maybe_ca_context = ca_comm.capture()  # type: ignore\n\n            from vllm._aiter_ops import rocm_aiter_ops\n",
                 "            if ca_comm is not None:\n                maybe_ca_context = ca_comm.capture()  # type: ignore\n"
                 "            if isinstance(self.device_communicator, CudaCommunicator):\n"
                 "                fi_pcie_ipc_ar_comm = self.device_communicator.fi_pcie_ipc_ar_comm\n"
                 "                if fi_pcie_ipc_ar_comm is not None:\n"
                 "                    maybe_fi_pcie_ipc_context = fi_pcie_ipc_ar_comm.capture()\n\n"
                 "            from vllm._aiter_ops import rocm_aiter_ops\n")
    replace_once(ps, "        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:\n            yield graph_capture_context\n",
                 "        with (torch.cuda.stream(stream), maybe_ca_context, maybe_fi_pcie_ipc_context, maybe_aiter_context):\n            yield graph_capture_context\n")
    commit(ps); print("patched parallel_state.py")

# 4) kernel_warmup.py
kw = files["kernel_warmup.py"]
if done(kw, "warmup_flashinfer_pcie_ipc_allreduce"): print("skip kernel_warmup.py (already patched)")
else:
    replace_once(kw, "    minimax_m3_msa_warmup(worker)\n\n    enable_flashinfer_autotune = (\n",
                 "    minimax_m3_msa_warmup(worker)\n\n    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:\n"
                 "        from vllm.distributed.device_communicators import flashinfer_pcie_ipc_all_reduce\n"
                 "        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)\n\n"
                 "    enable_flashinfer_autotune = (\n")
    commit(kw); print("patched kernel_warmup.py")

# 5) SM120: keep Sparse MLA off the dense-MHA prefill path (RTX 6000D / SM120)
#    adds `supports_dense_mha_prefill = False` right after `is_sparse = True` in the SM120 impl class
sm = files["flashinfer_mla_sparse_sm120.py"]
if done(sm, "supports_dense_mha_prefill = False"): print("skip flashinfer_mla_sparse_sm120.py (already patched)")
else:
    replace_once(sm, "    is_sparse = True\n", "    is_sparse = True\n    supports_dense_mha_prefill = False\n")
    commit(sm); print("patched flashinfer_mla_sparse_sm120.py")

# 6) new back-end modules
dc = root / "distributed/device_communicators"
for n in ("b12x_pcie_dma.py", "flashinfer_pcie_ipc_all_reduce.py"):
    src = Path(args.new_files_dir) / n
    if not src.exists(): sys.exit(f"missing {src}")
    if not args.dry_run: shutil.copy2(src, dc / n)
    print(("would install " if args.dry_run else "installed ") + str(dc / n))

if args.check:
    print("\n== compatibility check ==")
    for name, n in sorted(_ok.items()): print(f"  OK    {name}: {n} edit anchor(s) matched exactly once")
    for name, cnt, what in _miss: print(f"  MISS  {name}: anchor matched {cnt}x (need exactly 1): {what}")
    print("RESULT:", "COMPATIBLE" if not _miss else f"INCOMPATIBLE ({len(_miss)} anchor(s) need porting)")
    sys.exit(0 if not _miss else 1)
print("done" + (" (dry run)" if args.dry_run else ""))
