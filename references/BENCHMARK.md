# Benchmarking P/D against IFB and computing the gain

Goal: after the P/D deployment is verified, measure its real end-to-end performance and report the gain over a single-node
IFB (in-flight batching, "aggregate") baseline **under the same SLOs**. All numbers below are from the validated environment
(Dynamo 1.5.1 / vLLM 0.28.0, 8 x PCIe GPUs per node); they are a reproduction target for the same hardware, not a promise for other hardware.

## 1. Fairness conditions (all must hold, record them in the report)
- Same image / vLLM version, model (`GLM-5.2-NVFP4`), KV dtype (`fp8_e4m3`), MTP3, prefix-cache setting, max-num-batched-tokens (16384).
- IFB: one node, TP8 + EP, NCCL (no B12X / FlashInfer IPC). P/D: 1P1D (or your chosen ratio), B12X on P, FlashInfer IPC + PIECEWISE on D.
- Same workload and SLO for both: 58,944-token shared prefix + 6,592-token unique suffix (65,536 in, ~89.94% reuse), 1,024 out, `temperature=0`,
  `ignore_eos`, fixed seed, `num_prompts=100`, 1 warm-up request, request-rate `inf` with `--max-concurrency` as the ladder.
- **IFB server `--max-model-len` must be 66564.** The tokenizer round-trip can add 1 token to a 65,536-token prompt; the validated test produced
  INVALID runs with 65537 and 66560. Such files must be excluded (the analyzer skips names containing `invalid`, `smoke`, `warmup`).
- Clean KV / restart the workers between formal points (hook: `BETWEEN_POINTS_HOOK`) so earlier points do not pollute the cache.
- Same client path (same bench tool version); run both ladders back to back on the same hardware; no other load.

## 2. Procedure
1. Verify the P/D deployment first (`scripts/verify_deployment.py`; MTP acceptance >= IFB - 1 pp at C1).
2. P/D ladder: `templates/bench_ladder.sh pd`  (stops after `STOP_AFTER`=3 consecutive points that fail both gates).
3. Tear P/D down (or use other nodes), start the baseline: `templates/start_ifb.sh`, then `templates/bench_ladder.sh ifb`.
   The ladder must cover the boundary: at least 1-2 failing points above the last passing concurrency for **each** gate.
4. Analyze:
```bash
python3 scripts/analyze_benchmark.py --ifb <ifb result dir> --pd <pd result dir(s)> \
    --ifb-nodes 1 --pd-nodes 2 --gpus-per-node 8 --cp <C_P_A> <C_P_B> --cd <C_D> --out report.md
```
5. Put `report.md` next to the logs and cite it. If any gate has no passing point, extend the ladder downward (lower C) before concluding.

## 3. What the analyzer reports and how to read it
Per deployment and gate, the capacity point is the **highest concurrency that passes** (Gate A: TTFT p95 <= 10 s, user p99 > 30 tok/s;
Gate B: TTFT p50 < 5 s, user p50 > 55, user p99 > 25 tok/s). Never use the peak token/s of a failing point.

| Quantity | Formula | Meaning |
|---|---|---|
| Deployment gain | P/D req/s / IFB req/s - 1 | total throughput; P/D used more nodes |
| **Per-node / per-GPU gain** | (P/D req/s / P/D nodes) / (IFB req/s / IFB nodes) - 1; same for output tok/s per GPU | fair efficiency comparison; assumes IFB scales linearly with nodes |
| User speed | per-request 1/TPOT, TPOT = sum(itls)/(output_len-1); p50 = median, "p99" = slowest 1% (1st percentile) | decode experience; higher is better |
| TTFT delta | P/D TTFT - IFB TTFT at the respective capacity points (seconds) | P/D usually pays a little TTFT for KV transfer / queueing |
| Achievement | measured P/D req/s / min(C_P, C_D) | how much of the single-pool upper bound survives KV transfer, routing, queueing |

Always report **both** the deployment gain and the per-node gain; quoting only "3x" hides that P/D used 2 nodes.

## 4. Reproduction target (validated environment, N=100 per point, 0 failures)
| | Gate A | Gate B |
|---|---|---|
| IFB NCCL capacity | C5, 0.195 req/s, user p50/p99 45.6/31.3 | C4, 0.193 req/s, user p50/p99 57.8/45.2 |
| 1P1D B12X + FI IPC + PIECEWISE | C10, 0.586 req/s, user p50/p99 78.3/54.0 | C12, 0.647 req/s, user p50/p99 78.3/75.2 |
| Deployment gain / per-node gain | +200% / +50% | +236% / +68% |
| Achievement vs 1P1D bound (0.760 / 0.753) | 77% | 86% |
`scripts/analyze_benchmark.py` reproduces these exact numbers from the validated result files. On other hardware expect different absolute values;
what must hold is the **shape**: P/D passes the gates at clearly higher concurrency than IFB, user decode speed is higher, TTFT p50 is only modestly worse.

## 5. If the gain is much smaller than expected
1. MTP acceptance low on D? (PIECEWISE not active -> verify_deployment.py.)
2. Prefill saturated? (TTFT grows with concurrency while D is idle -> ratio, B12X dispatch, NCCL baseline.)
3. KV transfer/route bottleneck? (UCX device, NIXL side-channel IPs, cache hit; TTFT jitter with idle GPUs.)
4. Unfair baseline? (IFB max-model-len, different image/vLLM, prefix cache differences, restarts between points.)
5. Too few valid points (<100 requests, failures) or the ladder never reached the failing side of a gate.
Control experiment worth running: 1P1D with P and D both on NCCL. In the validated environment it was *below* IFB per node (-14% / -20%), i.e. the gain
comes from P/D **plus** the B12X / FlashInfer IPC back ends and PIECEWISE, not from disaggregation alone.

## 6. Report template
State: image tag + vLLM/Dynamo versions, hardware, ratio and node counts, workload, SLO definitions, ladder range, N per point, files excluded and why,
the capacity points per gate, deployment gain, per-node/per-GPU gain, user speed and TTFT deltas, achievement vs theory, MTP acceptance, caveats (e.g. IFB linear
scale-out assumption, single environment). Keep the raw JSON and logs.
