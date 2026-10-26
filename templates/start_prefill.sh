#!/usr/bin/env bash
# Prefill worker: B12X PCIe DMA AllReduce, MTP3, NIXL KV producer. Run on the PREFILL node.
# Start this BEFORE decode and wait for Ready (registered in Dynamo).
set -euo pipefail
source "${1:-$(dirname "$0")/common.env}"
mkdir -p "$RESULTS_DIR/prefill"; docker rm -f glm52-pd-prefill >/dev/null 2>&1 || true
CACHE_ARG="--no-enable-prefix-caching"; [[ "$PREFIX_CACHE" == on ]] && CACHE_ARG="--enable-prefix-caching"
docker run -d --name glm52-pd-prefill --network host --ipc host --shm-size 64g --gpus all \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_RESOURCE --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR":/models:ro -v "$RESULTS_DIR":/results \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" -e NATS_SERVER="$NATS_SERVER" -e DYN_DISCOVERY_BACKEND=etcd \
  -e DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT" \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_KV_IP" -e VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT" \
  -e UCX_NET_DEVICES="$UCX_NET_DEVICES" \
  -e VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=1 -e VLLM_B12X_DMA_MIN_BYTES=8388608 -e VLLM_B12X_DMA_MAX_BYTES=268435456 -e VLLM_B12X_DMA_FP8=ring \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_P2P_LEVEL="$PREFILL_NCCL_P2P_LEVEL" -e NCCL_SHM_DISABLE=1 \
  "$IMAGE" -lc "unset PROMETHEUS_MULTIPROC_DIR; exec python3 -m dynamo.vllm \
    --model $MODEL_PATH --served-model-name glm52 --disaggregation-mode prefill \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    --kv-events-config '{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:20081\",\"enable_kv_cache_events\":true}' \
    --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code --kv-cache-dtype fp8_e4m3 $CACHE_ARG \
    --max-model-len 66564 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
    --no-enable-flashinfer-autotune --spec-method mtp --spec-tokens 3 > /results/prefill/prefill.log 2>&1"
echo "prefill started; log: $RESULTS_DIR/prefill/prefill.log  (wait for registration before starting decode)"
