#!/usr/bin/env bash
# Concurrency ladder with `vllm bench serve` against a running deployment (P/D frontend or IFB server).
# usage: bench_ladder.sh <pd|ifb> [common.env]      (writes one JSON per point; stops early when both gates fail STOP_AFTER times in a row)
set -euo pipefail
MODE="${1:?pd|ifb}"; source "${2:-$(dirname "$0")/common.env}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# where `vllm bench serve` runs: any container/host that has the vllm CLI and can reach BASE_URL
BENCH_EXEC="${BENCH_EXEC:-docker exec $([[ $MODE == pd ]] && echo glm52-pd-prefill || echo glm52-ifb)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${FRONTEND_PORT:-8000}}"          # as seen from BENCH_EXEC
C_LIST="${C_LIST:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
N="${N:-100}"; WARMUPS="${WARMUPS:-1}"; SEED="${SEED:-6001}"
PREFIX_LEN="${PREFIX_LEN:-58944}"; SUFFIX_LEN="${SUFFIX_LEN:-6592}"; OUT_LEN="${OUT_LEN:-1024}"   # ~90% reuse, 65,536 in / 1,024 out
STOP_AFTER="${STOP_AFTER:-3}"; BETWEEN_POINTS_HOOK="${BETWEEN_POINTS_HOOK:-}"      # e.g. a script that restarts workers / clears KV
OUT_IN_CONTAINER="${OUT_IN_CONTAINER:-/results/bench/$MODE}"; OUT_ON_HOST="${OUT_ON_HOST:-$RESULTS_DIR/bench/$MODE}"
mkdir -p "$OUT_ON_HOST"
curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1 || echo "WARN: $BASE_URL/v1/models not reachable from this shell (ok if only reachable inside BENCH_EXEC)"
fails=0
for C in $C_LIST; do
  [[ -n "$BETWEEN_POINTS_HOOK" ]] && bash -c "$BETWEEN_POINTS_HOOK"
  f="$MODE-c$C-n$N.json"
  $BENCH_EXEC vllm bench serve --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model "$MODEL_PATH" --served-model-name glm52 --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len "$PREFIX_LEN" --prefix-repetition-suffix-len "$SUFFIX_LEN" --prefix-repetition-num-prefixes 1 \
    --prefix-repetition-output-len "$OUT_LEN" --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
    --ignore-eos --temperature 0 --seed "$SEED" --num-warmups "$WARMUPS" --metric-percentiles 99 \
    --save-result --save-detailed --result-dir "$OUT_IN_CONTAINER" --result-filename "$f" --disable-tqdm \
    > "$OUT_ON_HOST/${f%.json}.log" 2>&1 || { echo "C$C: bench command failed, see $OUT_ON_HOST/${f%.json}.log"; break; }
  a=0; b=0
  python3 "$HERE/scripts/analyze_benchmark.py" check "$OUT_ON_HOST/$f" --gate A && a=1 || true
  python3 "$HERE/scripts/analyze_benchmark.py" check "$OUT_ON_HOST/$f" --gate B >/dev/null && b=1 || true
  if [[ $a == 0 && $b == 0 ]]; then fails=$((fails+1)); else fails=0; fi
  [[ $fails -ge $STOP_AFTER ]] && { echo "both gates failed $STOP_AFTER points in a row: stop"; break; }
done
echo "done. Analyze with: python3 $HERE/scripts/analyze_benchmark.py --ifb <ifb dir> --pd <pd dir> ..."
