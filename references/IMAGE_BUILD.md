# Building / rebuilding the "dual PCIe" runtime image

> Preferred path now: `python3 scripts/auto_build.py run` (uses `Dockerfile.dual`, reproducible, anchor-checked). The docker-commit procedure below is how the validated 1.5.1 image was originally made and remains available as `--method commit`.

Target: `dynamo-vllm-runtime:<dynamo>-vllm<NNN>-pcie-dual` = official Dynamo vLLM runtime + two PCIe AllReduce back ends
(B12X PCIe DMA for Prefill, FlashInfer PCIe IPC for Decode) + one SM120 compatibility patch.

Current validated build (recorded by the image builder):
- Base: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1` (Dynamo 1.5.1, vLLM 0.28.0)
- Result: image ID `00bceaa0c57fe8774346dbd729b9d84f48b9b3fee8ed962d653a49164d795d33`, ~45.3 GB, created 2026-10-09 12:51 CST
- **Provenance caveat:** built by patching a running container and `docker commit`, NOT from a Dockerfile. It is not reproducible from source alone. Prefer converting the steps below into a Dockerfile, and keep the patch files in `patches/` of this skill.

## Procedure (as performed for 1.5.1 / vLLM 0.28.0)
1. Start a root work container from the base image, `--entrypoint /bin/bash`, `--gpus all --network host --ipc host`, mounting `/models` and a patch dir; `sleep infinity`.
2. Install comms dependencies with `pip install --no-cache-dir --no-deps`:
   - `b12x==1.3.0`
   - `flashinfer-python==0.7.0.post1`
   - also `flashinfer-cubin==0.7.0.post1`, `nccl-extensions==0.1.0`, `nvidia-cudnn-frontend==1.30.0`
   - `flashinfer-cubin` and `flashinfer_jit_cache` were copied from an already-validated vLLM 0.31 container (not pip-installed). Record the source container when you redo this.
3. Add two back-end files under `vllm/distributed/device_communicators/`:
   - `b12x_pcie_dma.py` - own vLLM adapter that calls `b12x.comm.pcie.DmaAllReduce`.
   - `flashinfer_pcie_ipc_all_reduce.py` - ported from the validated vLLM 0.31 environment to 0.28.
4. Modify these vLLM files:
   - `vllm/envs.py`: add `VLLM_ALLREDUCE_USE_B12X_PCIE_DMA`, `VLLM_B12X_DMA_MIN_BYTES`, `VLLM_B12X_DMA_MAX_BYTES`, `VLLM_B12X_DMA_FP8`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`.
   - `vllm/distributed/device_communicators/cuda_communicator.py`: init `B12xPcieDmaAllReduce` and `FlashInferPcieIpcAllReduce`; dispatch order before NCCL: `B12X_PCIE_DMA`, `FLASHINFER_PCIE_IPC`, then `NCCL_SYMM_MEM / FLASHINFER / CUSTOM / PYNCCL`; an unsupported tensor returns `None` so the next back end handles it; enable only on real `tp:*` groups (never other groups); call `close()` / `destroy()` on teardown.
   - `vllm/distributed/parallel_state.py`: add FlashInfer PCIe IPC `capture()` to the CUDA Graph capture context.
   - `vllm/model_executor/warmup/kernel_warmup.py`: when FlashInfer PCIe IPC is on, call `warmup_flashinfer_pcie_ipc_allreduce(worker)`.
   - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`: set `supports_dense_mha_prefill = False` (RTX 6000D / SM120 sparse MLA must not take the dense-MHA prefill path).
5. Clear the vLLM compile cache: `rm -rf /home/dynamo/.cache/vllm`.
6. `docker commit <work-container> dynamo-vllm-runtime:<dynamo>-vllm<NNN>-pcie-dual`.

Role env vars (what the templates set): Prefill `VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=1`, `VLLM_B12X_DMA_MIN_BYTES=8388608`, `VLLM_B12X_DMA_MAX_BYTES=268435456`, `VLLM_B12X_DMA_FP8=ring`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0`; Decode `VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=0`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`.

Patch artifacts are stored in `patches/vllm-0.28/` (copied from the original image builder's patch directory):
- `apply_all.py` - **use this.** Idempotent single-step patcher; byte-identical output to `final/` (verified against `original/`); stages edits per file so a failed match never leaves a half-patched file; `--dry-run`, `--root`, `--backup-dir`.
- `apply_patch.py` + `apply_patch_remaining.py` - the builder's original two scripts. They are meant to be run **together, in that order**: `apply_patch.py` aborts at its `cuda_communicator.py` destroy step (anchor matches 3 places) and `apply_patch_remaining.py` finishes destroy + `parallel_state.py` + `kernel_warmup.py`. They are non-transactional and not re-runnable. Kept for provenance.
- The SM120 one-liner (`supports_dense_mha_prefill = False`) was a manual edit in the original build; `apply_all.py` performs it.
- `original/` = unmodified vLLM 0.28.0 files, `final/` = patched files plus the two new back-end modules, `prefill.env` / `decode.env` = role env vars.

## Upgrading to a newer Dynamo base (e.g. 1.6.0 = vLLM 0.31.0)
`scripts/find_latest_image.py` reports `UPSTREAM_NEWER=Dynamo:<ver>` when NGC has a newer official image. Then:
1. Pull `OFFICIAL_LATEST` and check what it already contains. The FlashInfer PCIe-IPC back end was ported *from* a vLLM 0.31 environment, so a 0.31-based image may already ship it natively - check `vllm/envs.py` for `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` and `cuda_communicator.py` for `FlashInferPcieIpcAllReduce` before patching anything.
2. Re-port only what is missing (likely the B12X adapter and env vars); `apply_all.py` uses exact-string anchors for vLLM 0.28.0 and will refuse (without modifying anything) on a different version - port by intent using `original/` vs `final/` as the reference diff.
3. Re-check the SM120 `supports_dense_mha_prefill` patch is still needed/located in the same file.
4. Run `scripts/build_dual_image.sh` (or your Dockerfile), then the full validation gate in SKILL.md and add a `VALIDATED:` line. Re-verify PIECEWISE on Decode: new vLLM versions can change graph and MTP behaviour.
