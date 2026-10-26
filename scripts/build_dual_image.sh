#!/usr/bin/env bash
# Rebuild the dual-PCIe runtime image the way the 1.5.1 / vLLM 0.28.0 image was made
# (patch a running container, then docker commit). See references/IMAGE_BUILD.md.
# usage: BASE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1 TARGET=dynamo-vllm-runtime:1.5.1-vllm028-pcie-dual \
#        PATCH_DIR=/path/to/patches/vllm-0.28 scripts/build_dual_image.sh
# PATCH_DIR = patches/vllm-0.28 (contains apply_all.py + final/). apply_all.py is idempotent and edits/adds the vLLM files
# listed in references/IMAGE_BUILD.md. These patches are tied to ONE vLLM version (0.28.0).
# optional: VLLM_ROOT (default /usr/local/lib/python3.12/dist-packages/vllm inside the image)
set -euo pipefail
: "${BASE:?set BASE (official nvcr.io/nvidia/ai-dynamo/vllm-runtime:<ver>)}"
: "${TARGET:?set TARGET (dynamo-vllm-runtime:<dynamo>-vllm<NNN>-pcie-dual)}"
: "${PATCH_DIR:?set PATCH_DIR (dir with apply_patch.py)}"
B12X_VER="${B12X_VER:-1.3.0}"; FI_VER="${FI_VER:-0.7.0.post1}"
CUDNN_FE_VER="${CUDNN_FE_VER:-1.30.0}"; NCCL_EXT_VER="${NCCL_EXT_VER:-0.1.0}"
# Optional: a validated container that already has flashinfer-cubin / flashinfer_jit_cache (the 1.5.1 build copied them from vLLM 0.31)
CUBIN_SRC_CONTAINER="${CUBIN_SRC_CONTAINER:-}"
WORK="${WORK:-glm52-dual-build}"
[[ -f "$PATCH_DIR/apply_all.py" && -d "$PATCH_DIR/final" ]] || { echo "need $PATCH_DIR/apply_all.py and $PATCH_DIR/final/" >&2; exit 1; }
docker rm -f "$WORK" >/dev/null 2>&1 || true
docker run -d --name "$WORK" --user root --gpus all --network host --ipc host --entrypoint /bin/bash \
  -v "$PATCH_DIR":/patch:ro -v /models:/models "$BASE" -lc 'sleep infinity'
docker exec "$WORK" python3 -m pip install --no-cache-dir --no-deps "b12x==$B12X_VER" "flashinfer-python==$FI_VER" \
  "flashinfer-cubin==$FI_VER" "nccl-extensions==$NCCL_EXT_VER" "nvidia-cudnn-frontend==$CUDNN_FE_VER" \
  || echo "WARN: some wheels failed (cubin/jit-cache may need copying from a validated container)"
if [[ -n "$CUBIN_SRC_CONTAINER" ]]; then
  echo "copy flashinfer_jit_cache / cubin from $CUBIN_SRC_CONTAINER manually (site-packages paths differ per image)"; fi
docker exec "$WORK" python3 /patch/apply_all.py --backup-dir /root/vllm-original ${VLLM_ROOT:+--root "$VLLM_ROOT"}
docker exec "$WORK" python3 -c "import vllm.distributed.device_communicators.b12x_pcie_dma, vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce, vllm.envs as e; print('import ok; B12X env:', e.VLLM_ALLREDUCE_USE_B12X_PCIE_DMA, 'FI IPC env:', e.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC)"
docker exec "$WORK" rm -rf /home/dynamo/.cache/vllm
echo "Patched. Run a quick import/launch smoke test in the work container BEFORE committing."
read -r -p "commit $WORK as $TARGET ? [y/N] " a; [[ "$a" == y ]] || exit 0
docker commit "$WORK" "$TARGET"; docker image ls "$TARGET"
echo "Next: run the validation gate in SKILL.md, then add a VALIDATED: line to references/validated-images.md"
