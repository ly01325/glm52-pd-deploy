#!/usr/bin/env bash
# IFB (in-flight batching / aggregate) BASELINE on ONE node: TP8 + EP, MTP3, NCCL (no B12X, no FlashInfer IPC).
# Used to benchmark P/D against. Must run the SAME image/vLLM version, model, KV dtype, MTP and prefix-cache settings as P/D.
# UNVERIFIED template: derived from the validated IFB server's logged engine args; check `vllm serve --help` on your image.
set -euo pipefail
source "${1:-$(dirname "$0")/common.env}"
mkdir -p "$RESULTS_DIR/ifb"; docker rm -f glm52-ifb >/dev/null 2>&1 || true
CACHE_ARG="--no-enable-prefix-caching"; [[ "$PREFIX_CACHE" == on ]] && CACHE_ARG="--enable-prefix-caching"
# NOTE: --max-model-len 66564 is required: the tokenizer round-trip can add 1 token to a 65,536-token prompt, and
# max-model-len 65537 / 66560 produced INVALID runs in the validated test.
docker run -d --name glm52-ifb --network host --ipc host --shm-size 64g --gpus all \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_RESOURCE --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR":/models:ro -v "$RESULTS_DIR":/results \
  -e VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=0 -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0 \
  ${IFB_EXTRA_DOCKER_ARGS:-} \
  "$IMAGE" -lc "exec vllm serve $MODEL_PATH --served-model-name glm52 --host 0.0.0.0 --port ${FRONTEND_PORT:-8000} \
    --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code --kv-cache-dtype fp8_e4m3 $CACHE_ARG \
    --max-model-len 66564 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
    --no-enable-flashinfer-autotune --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 > /results/ifb/ifb.log 2>&1"
echo "IFB baseline started; log: $RESULTS_DIR/ifb/ifb.log"
