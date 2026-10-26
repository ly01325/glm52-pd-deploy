# Porting the dual-PCIe patch to a new vLLM series

Use when `scripts/auto_build.py` reports "no patch set" (exit 3) or `check_patch_compat.py` reports MISS anchors.
The reference implementation is `patches/vllm-0.28/` (`original/` = pristine vLLM 0.28.0 files, `final/` = patched).
Port **by intent, not by diff**: re-create the behaviour below in the new vLLM files, then encode it as anchored edits in a
new `patches/vllm-<series>/apply_all.py`. Nothing here has been run against 0.31 yet - treat every "expect" as a hypothesis to check.

## 0. Start
```bash
python3 scripts/ngc_info.py table                       # tag -> vLLM mapping, from registry labels (no pull)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>  # if 401: docker login nvcr.io  (user $oauthtoken, pass = NGC API key)
python3 scripts/probe_base_image.py <image>              # versions + which parts are ALREADY native
python3 scripts/scaffold_port.py --image <image>         # patches/vllm-<series>/ with original/, final/, DRIFT.md, PORTING_TODO.md
```
`DRIFT.md` shows per-file drift against the 0.28 originals and which anchors of the copied `apply_all.py` no longer match.

## 1. Decide what is actually missing (native support)
The FlashInfer PCIe-IPC module was originally ported **from a vLLM 0.31 environment**, and the B12X adapter was written by us. So on a 0.31-based image
expect (hypothesis): FlashInfer PCIe IPC may already exist natively or at least have the same shape; the B12X adapter does not.
`probe_base_image.py` prints `FEATURE_STATE` and `NATIVE_*`:
- `none` -> port everything below.
- `partial-native` / `full-native` -> do **not** run the 0.28 `apply_all.py` (it would add duplicate attributes/branches). Port only what is missing,
  reuse the native env var names/class names, and keep our env-var contract for the B12X pieces (see section 4).
Check that the runtime contract still holds: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` => log `Using ['FLASHINFER_PCIE_IPC', 'PYNCCL'] ... group 'tp:0'`.

## 2. Intent of each edit (what must be true after porting)
| File (under vllm/) | Required behaviour | How to find the place in a new version |
|---|---|---|
| `envs.py` | 5 new env vars: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` (bool, 0), `VLLM_ALLREDUCE_USE_B12X_PCIE_DMA` (bool, 0), `VLLM_B12X_DMA_MIN_BYTES` (int, 8 MiB), `VLLM_B12X_DMA_MAX_BYTES` (int, 256 MiB), `VLLM_B12X_DMA_FP8` (str, "ring"). Both a type-hint block entry and an `environment_variables` dict entry. | grep `VLLM_ALLREDUCE_USE_FLASHINFER` in `envs.py` and add next to it |
| `distributed/device_communicators/cuda_communicator.py` | (a) The tp-only gate must match the real `tp:*` group name (`unique_name.split(":")[0] == "tp"`), not a substring test. (b) Compute `use_b12x_pcie_dma_allreduce` / `use_flashinfer_pcie_ipc_allreduce` = env flag and not `VLLM_BATCH_INVARIANT`, False for non-tp groups. (c) Construct `B12xPcieDmaAllReduce(group=device_group, device)` and `FlashInferPcieIpcAllReduce(group=device_group, tune_group=cpu_group, device)` when `world_size > 1 and device_group is not None`. (d) In `all_reduce()` dispatch, **before the NCCL/PYNCCL fallback and before the `fi_ar_comm` branch**: try B12X first (`out = comm.all_reduce(x)`; if `None` fall through), then FlashInfer IPC (`should_use(x)` -> `all_reduce(x)`). Unsupported tensors must return/skip so later back ends handle them. (e) Add both names to the logged back-end list in dispatch order: `B12X_PCIE_DMA`, `FLASHINFER_PCIE_IPC`, then the existing ones, and append them to `enabled_ar_backends` when enabled and not `disabled`. (f) `destroy()`: `fi_pcie_ipc_ar_comm.destroy()` and `b12x_pcie_dma_ar_comm.close()`. | grep `fi_ar_comm`, `enabled_ar_backends`, `all_potential_ar_backends`, `"tp" not in unique_name`, `def destroy` |
| `distributed/parallel_state.py` | In `graph_capture`, enter `fi_pcie_ipc_ar_comm.capture()` (if present) together with the existing capture contexts so IPC buffers are valid inside CUDA-graph capture. | grep `maybe_ca_context`, `torch.cuda.stream(stream)` |
| `model_executor/warmup/kernel_warmup.py` | When `envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, call `warmup_flashinfer_pcie_ipc_allreduce(worker)` before graph capture / FlashInfer autotune. | grep `minimax_m3_msa_warmup(worker)` / `enable_flashinfer_autotune` |
| `v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `supports_dense_mha_prefill = False` on the SM120 sparse-MLA impl class (RTX 6000D / SM120 must not take the dense-MHA prefill path). If the new version already sets/handles this differently, keep the effect, not the line. | the class next to `is_sparse = True` |
| new `distributed/device_communicators/b12x_pcie_dma.py` | Adapter over `b12x.comm.pcie.DmaAllReduce(exchange_group, device, max_bytes, fp8)`. Contract: `.disabled`, `.all_reduce(t) -> Tensor|None` (None = not handled), `.close()`. World size in (2,4,8), cuda only. Log `B12x PCIe DMA all-reduce enabled (world=..., wire=..., bytes=min..max)` once (verify_deployment.py greps this). | copy from `final/`; check b12x API |
| new `.../flashinfer_pcie_ipc_all_reduce.py` | Wrapper over `flashinfer.comm.PcieIpcAllReduceWorkspace`. Contract: `.disabled`, `.setup(hidden_dim, dtype, capture_sizes, tune_cache)`, `.should_use(t)`, `.all_reduce(t)`, `.capture()` (context manager), `.destroy()`, plus module function `warmup_flashinfer_pcie_ipc_allreduce(worker)`. | copy from `final/` or reuse the native one |

## 3. External APIs the two new modules depend on - re-verify each on the new version
`probe_base_image.py` reports the ones it can grep (`MISSING_VLLM_API=`). Also check by hand:
- vLLM: `vllm.distributed.parallel_state.in_the_same_node_as`; `vllm.model_executor.warmup.flashinfer_autotune_cache.resolve_flashinfer_autotune_file`; `vllm_config.parallel_config.use_ubatching`; `vllm_config.compilation_config.cudagraph_capture_sizes`; `model_config.get_hidden_size()`; `logger.info_once / warning_once`; `torch.accelerator.synchronize` (needs a recent torch).
- FlashInfer: `flashinfer.comm.PcieIpcAllReduceWorkspace` with methods `rebind_stream, tune, prepare, supports, all_reduce, destroy, capture-related` and ctor args `(group, max_numel, dtype, tune_batches, tune_cache)`. Do not downgrade a base image's newer flashinfer-python (auto_build.py keeps it); our pins are for 0.28 (`flashinfer-python==0.7.0.post1`, `b12x==1.3.0`).
- b12x: `b12x.comm.pcie.DmaAllReduce`, attributes `wire_mode, min_bytes, max_bytes`, methods `should_allreduce, all_reduce, close`.
If a symbol moved, adapt the module; do not add shims that silently change numerics or dispatch.

## 4. Behavioural contract that must survive any port (checked by verify_deployment.py / acceptance)
- Prefill: `VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=1`, `VLLM_B12X_DMA_MIN_BYTES=8388608`, `VLLM_B12X_DMA_MAX_BYTES=268435456`, `VLLM_B12X_DMA_FP8=ring`, FlashInfer IPC off -> TP group `Using ['B12X_PCIE_DMA', 'PYNCCL']`.
- Decode: B12X off, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` -> TP group `Using ['FLASHINFER_PCIE_IPC', 'PYNCCL']`.
- EP group stays on PYNCCL; enabled **only** on `tp:*` groups.
- CUDA-graph capture works with PIECEWISE (and must not break MTP acceptance in P/D - re-run the acceptance gate; a new vLLM can change graph/MTP behaviour).

## 5. Make the port mechanical and checkable
1. Edit `patches/vllm-<series>/final/*.py` to the target state; `python3 -m py_compile final/*.py`.
2. Edit that set's `apply_all.py` anchors (exact strings from `original/`) until, on a tree containing only `original/` files:
   `python3 apply_all.py --root <tree> --check` -> `COMPATIBLE`, a real apply reproduces `final/` byte-for-byte (`cmp`), and a second apply is a no-op.
   Anchors must match exactly once; never loosen an edit to "replace all".
3. Fill `TARGET.json` (`vllm`, `vllm_series`, `python`, `vllm_site_packages_root`, pip pins; remove the `SCAFFOLD` status line).
4. `python3 scripts/check_patch_compat.py <image> --patch-set vllm-<series>` -> COMPATIBLE against the real image.
5. `python3 scripts/auto_build.py run --base-tag <dynamo tag>`; it builds with `Dockerfile.dual` and smoke-imports both modules.
6. GPU acceptance (SKILL.md steps 5-7). Only after it passes: add a `VALIDATED:` line, set `TARGET.json.validated_on`.

## 6. When to stop and fall back
After a reasonable effort (anchors fixed but import/smoke fails, a b12x/FlashInfer API is gone, or acceptance regresses), stop. Report what failed and
build on the newest base that already has a patch set: `auto_build.py run --policy newest-patched`. Never mark an unvalidated image as recommended.
