#!/usr/bin/env bash
# Decode worker: FlashInfer PCIe IPC AllReduce, MTP3, NIXL KV consumer, CUDA graph PIECEWISE (mandatory).
# Run on the DECODE node AFTER prefill is registered and Ready.
set -euo pipefail
source "${1:-$(dirname "$0")/common.env}"
mkdir -p "$RESULTS_DIR/decode"; docker rm -f glm52-pd-decode >/dev/null 2>&1 || true
CACHE_ARG="--no-enable-prefix-caching"; [[ "$PREFIX_CACHE" == on ]] && CACHE_ARG="--enable-prefix-caching"
docker run -d --name glm52-pd-decode --network host --ipc host --shm-size 64g --gpus all \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_RESOURCE --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR":/models:ro -v "$RESULTS_DIR":/results \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" -e NATS_SERVER="$NATS_SERVER" -e DYN_DISCOVERY_BACKEND=etcd \
  -e DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$DECODE_KV_IP" -e VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT" \
  -e UCX_NET_DEVICES="$UCX_NET_DEVICES" \
  -e VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=0 -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0 \
  -e NCCL_P2P_LEVEL="$DECODE_NCCL_P2P_LEVEL" -e NCCL_NET_DISABLE_INTRA=1 -e NCCL_PROTO=LL \
  "$IMAGE" -lc "unset PROMETHEUS_MULTIPROC_DIR; exec python3 -m dynamo.vllm \
    --model $MODEL_PATH --served-model-name glm52 --disaggregation-mode decode \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code --kv-cache-dtype fp8_e4m3 $CACHE_ARG \
    --max-model-len 66564 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
    --spec-method mtp --spec-tokens 3 --no-enable-flashinfer-autotune \
    -cc.cudagraph_mode=PIECEWISE > /results/decode/decode.log 2>&1"
echo "decode started; log: $RESULTS_DIR/decode/decode.log"
