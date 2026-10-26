#!/usr/bin/env bash
# Dynamo Frontend/Router. Requires etcd + NATS already running (control plane).
set -euo pipefail
source "${1:-$(dirname "$0")/common.env}"
docker rm -f glm52-frontend >/dev/null 2>&1 || true
docker run -d --name glm52-frontend --network host \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" -e NATS_SERVER="$NATS_SERVER" -e DYN_DISCOVERY_BACKEND=etcd \
  "$IMAGE" -lc "exec python3 -m dynamo.frontend --http-port ${FRONTEND_PORT}"
echo "frontend started on :${FRONTEND_PORT}"
