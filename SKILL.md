---
name: glm52-pd-deploy
description: Deploy, validate and size GLM-5.2-NVFP4 prefill/decode (P/D) disaggregated serving on vLLM + Dynamo (B12X PCIe DMA on Prefill, FlashInfer PCIe IPC on Decode, CUDA Graph PIECEWISE, NIXL/UCX KV transfer). Use when asked to deploy GLM 5.2 P/D, pick the P:D ratio, find the latest Dynamo runtime image, build or port the dual-PCIe image on a new Dynamo/vLLM version, verify a deployment from logs, benchmark P/D against an IFB baseline and compute the throughput/efficiency gain, or debug low MTP acceptance / no gain from adding Decode nodes.
---

# GLM-5.2 P/D deployment

Validated stack: GLM-5.2-NVFP4, 8 GPU per instance, TP8 + EP, vLLM 0.28, Dynamo 1.5.1, MTP3, FP8 KV cache.
Authoritative long-form guide: `references/DEPLOYMENT_HANDBOOK.md` (section numbers below refer to it).
Result you should reproduce: 1P1D vs single-node IFB NCCL = about 3.0x (Gate A) / 3.36x (Gate B) deployment throughput, MTP acceptance about 95.6%.

## Hard rules (do not negotiate these away)

1. **Decode MUST use `-cc.cudagraph_mode=PIECEWISE`.** With the default FULL graph, MTP acceptance collapses to ~14-15% (vs ~95%). Cause: FULL graph bypasses the per-request external-KV wait/restore around sparse-MLA attention.
2. **Different AllReduce back ends per role:** Prefill = B12X PCIe DMA, Decode = FlashInfer PCIe IPC. Never enable both on one role.
3. **Start order:** etcd + NATS -> Frontend -> Prefill -> wait until Prefill is registered/Ready -> Decode. Never cold-start P and D in parallel.
4. **Never copy IPs, NIC/HCA names, `NCCL_P2P_LEVEL`, `NCCL_SHM_DISABLE`, `UCX_NET_DEVICES` from another cluster.** Derive them from this machine's topology (Handbook s4).
5. **Size by capacity, not GPU count:** N_D/N_P ~ C_P/C_D, measured per SLO, then validate neighbouring integer ratios end to end.
6. **Turning Prefix Cache off needs `--no-enable-prefix-caching`.** Omitting the enable flag does not disable it in vLLM 0.28.
7. Claims like "B12X/FlashInfer enabled" need **log evidence**; run `scripts/verify_deployment.py`.

## Inputs to collect from the user (ask, do not guess)

Prefill/Decode node addresses, control-plane host (etcd/NATS), business-network IPs for NIXL side channel, UCX device (`ucx_info -d`, `ibdev2netdev`), model path, results dir, SLO (default Gate A/B below), expected traffic shape (default 64K in / 1K out, ~90% reuse), image registry details (see Image step).

## Workflow

### 0. Get the runtime image (always first): find the newest, build on it, or fall back
Lineage: official base = NVIDIA NGC `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<dynamo-ver>` (public; plain tags `1.5.1`, `1.6.0`). We add B12X PCIe-DMA + FlashInfer PCIe-IPC AllReduce on top and name the result `dynamo-vllm-runtime:<dynamo>-vllm<NNN>-pcie-dual`. The patch is tied to ONE vLLM series (`patches/vllm-<series>/`, currently only 0.28). The registry itself tells you which vLLM each Dynamo tag ships (image labels), so no pull is needed to decide.

```bash
python3 scripts/find_latest_image.py        # do we already have a usable derived image? (local/registry + validated list)
python3 scripts/ngc_info.py table           # newest official Dynamo tags -> bundled vLLM, CUDA, date, patch-set status
python3 scripts/auto_build.py plan          # what a build on the NEWEST official image would do (network only)
```
Decision tree (act on the `KEY=VALUE` lines / exit codes):
1. `find_latest_image.py` prints `RECOMMENDED_IMAGE` and `UPSTREAM_NEWER=none` -> use it. Done.
2. `UPSTREAM_NEWER=Dynamo:<ver>` or no derived image (exit 2) -> run `python3 scripts/auto_build.py run` (policy `newest`: pull the newest official image, probe it, check the patch, build with `Dockerfile.dual`, smoke-test, write `build-reports/<tag>.md`).
   - exit 0 -> image built. It is **not validated**; go to the validation gate below.
   - exit 3 "patch must be ported" (newest vLLM series has no patch set, or anchors do not match) -> `auto_build.py` already ran `scaffold_port.py` and created `patches/vllm-<series>/{original,final,apply_all.py,DRIFT.md,PORTING_TODO.md}`. Port the patch by following `references/PORTING_GUIDE.md` (check native support first with `probe_base_image.py`), until `check_patch_compat.py <image> --patch-set vllm-<series>` says COMPATIBLE, then re-run `auto_build.py run --base-tag <tag>`.
   - exit 4 -> build/smoke failed; read the error (pip wheel for this CUDA, anchor mismatch, network). Fix once or twice; do not loop.
   - exit 2 -> environment (no docker / no NGC access / pull 401: `docker login nvcr.io`, user `$oauthtoken`, password = NGC API key).
3. Porting fails or runtime acceptance regresses -> fall back: `python3 scripts/auto_build.py run --policy newest-patched` (newest Dynamo tag whose vLLM already has a patch set, today 1.5.1 / vLLM 0.28). Tell the user the newest image could not be adopted and why.
Never swap to an unvalidated image for production, and never silently change vLLM/Dynamo versions.

Helpers: `scripts/probe_base_image.py <image>` (vLLM, packages, native features), `scripts/check_patch_compat.py <image>` (exact-anchor check without running the image; `--from-dir` offline), `scripts/scaffold_port.py`, `scripts/build_dual_image.sh` (legacy docker-commit path, `--method commit`). Details: `references/IMAGE_BUILD.md`, `references/PORTING_GUIDE.md`.
Derived-image sources for `find_latest_image.py` come from env vars: `GLM52_IMAGE_REPO`, `GLM52_IMAGE_SUFFIX` (default `pcie-dual`), `GLM52_REGISTRY_URL` + `GLM52_REGISTRY_REPO` (+ `GLM52_REGISTRY_TOKEN` or `GLM52_REGISTRY_USER/PASS`), `GLM52_TAGS_CMD`, `GLM52_SKOPEO_REF`. NGC lookups need no configuration.

**Validation gate for a new image** (then append a `VALIDATED:` line to `references/validated-images.md`): `verify_deployment.py` passes on real P and D logs; C1, N=100, 1024-token run has 100% completion and MTP acceptance >= IFB reference minus 1 pp (IFB: 94.54%); Gate A/B boundary points re-run at N=100; no Traceback/CUDA error/OOM. A new Dynamo/vLLM base can change CUDA-graph, MTP and AllReduce behaviour, so the PIECEWISE and dispatch checks must be repeated. The build/smoke steps above only prove the image imports; they prove nothing about performance or correctness.

### 1. Freeze versions and record topology (Handbook s3, s4.2)
Record GPU/driver/CUDA/NCCL, vLLM/Dynamo/FlashInfer/B12X/NIXL/UCX versions, model checksum, TP/EP, KV dtype, MTP, graph mode, SLO. Collect `nvidia-smi topo -m`, `nvidia-smi topo -p2p r|w`, `numactl -H`, `ibdev2netdev`, `ibstat`, `ucx_info -d`. Confirm GPU P2P works and the UCX device is Active and on the business network (also inside the container).

### 2. NCCL baseline (Handbook s4.3-4.4)
Run `nccl-tests all_reduce_perf -b 8 -e 512M -f 2 -g 8` with defaults first, then A/B one explicit variable at a time. Accept only if: intended HCA used (no Socket fallback), stable within ~3%, no warnings/timeouts. EP traffic still goes through PYNCCL, so this baseline matters even with B12X/FlashInfer.

### 3. Single-pool capacity (Handbook s5)
Measure `C_P` (Prefill at the SLO) and `C_D` (Decode at the SLO) independently with identical SLO and workload.
- Gate A: TTFT p95 <= 10 s, user output p99 > 30 tok/s.
- Gate B: TTFT p50 < 5 s, user output p50 > 55 and p99 > 25 tok/s.
- Take the highest *stable request rate that meets the SLO*, not peak token/s. Pre-warm; 100 requests per formal point; clean KV between points.
- Decode test: pre-fill the full KV, fixed 1024 output, drop first/last 16 of 100 completions for steady req/s.
Reference values on the validated hardware: C_P = 0.7603 / 0.7526 req/s (A/B), C_D = 0.9413 req/s.

### 4. Ratio
```bash
python3 scripts/compute_pd_ratio.py --cp <C_P> --cd <C_D> --headroom 0.2
```
Reference result: D/P ~ 0.8, so 1P1D for small scale, 5P:4D to start at larger scale; Prefill is the bottleneck, adding Decode does not help. Keep 15-30% headroom on the bottleneck pool. Treat the output as a starting point and test neighbouring integer ratios end to end.

### 5. Deploy (Handbook s7)
Copy `templates/common.env.example` to `templates/common.env`, fill it in, then:
```bash
templates/start_frontend.sh   # control node (etcd + NATS must already be up)
templates/start_prefill.sh    # prefill node; wait until registered/Ready
templates/start_decode.sh     # decode node; only after prefill is Ready
```
The templates mirror the validated launch flags (TP8, `--enable-expert-parallel`, `fp8_e4m3` KV, `--max-model-len 66564`, `--max-num-batched-tokens 16384`, `--gpu-memory-utilization 0.92`, `--spec-method mtp --spec-tokens 3`, NixlConnector, Prefill `kv_events` publisher). Edit values, not structure.

### 6. Verify from logs
```bash
python3 scripts/verify_deployment.py --prefill-log <prefill.log> --decode-log <decode.log> [--mtp-acceptance <pct>]
```
All required checks must PASS (B12X enabled on Prefill; `FLASHINFER_PCIE_IPC` on Decode TP group; `PIECEWISE`; MTP; NIXL; no errors). EP group showing `PYNCCL` is expected. Then send one smoke request through the frontend and confirm it went P -> NIXL KV -> D.

### 7. End-to-end acceptance and go-live (Handbook s8)
Step the concurrency ladder for each candidate ratio (cold and target-reuse), N=100 per formal point with 100% completion, restart or clear KV between points, one soak test and one worker-restart test. Plan capacity from end-to-end results (they sit below the single-pool upper bound because of KV transfer, routing, queueing). Go-live checklist is in Handbook s8.3.

### 8. Benchmark P/D vs IFB and compute the gain (BENCHMARK.md)
Final deliverable: the measured gain of the P/D deployment over a single-node IFB NCCL baseline under the same SLOs.
1. Same image, model, KV dtype, MTP3, prefix cache; workload 58,944 shared prefix + 6,592 suffix -> 65,536 in / 1,024 out; N=100 per point; 1 warm-up; fixed seed. **IFB must use `--max-model-len 66564`** (65537/66560 gave invalid runs).
2. P/D ladder: `templates/bench_ladder.sh pd`. Baseline: `templates/start_ifb.sh` then `templates/bench_ladder.sh ifb`. Cover at least 1-2 failing concurrencies above each gate's last pass; clean KV / restart between formal points.
3. Compute and write the report:
```bash
python3 scripts/analyze_benchmark.py --ifb <ifb dir> --pd <pd dir> --ifb-nodes 1 --pd-nodes 2 --gpus-per-node 8 --cp <C_P_A> <C_P_B> --cd <C_D> --out report.md
```
4. Report **both** deployment gain (P/D req/s / IFB req/s - 1) and the fair **per-node / per-GPU gain**, plus user-speed and TTFT deltas and the achievement vs the 1P1D bound. Reproduction target on the validated hardware: +200% / +50% (Gate A) and +236% / +68% (Gate B) - see `references/BENCHMARK.md` s4. If the gain is far lower, follow its s5 checklist before concluding. Never quote a failing point's peak.

## Diagnose fast

| Symptom | Likely cause -> action |
|---|---|
| Adding D gives no throughput | Prefill saturated -> check P queue/TTFT, add P |
| MTP acceptance ~14-15% | Decode on FULL graph -> `-cc.cudagraph_mode=PIECEWISE`, restart |
| FI/B12X configured, no change | not dispatched -> read the `Using [...] all-reduce backends` log lines |
| TP faster, overall still slow | EP on PYNCCL -> check `ep:0` log and NCCL baseline |
| TTFT jitter, GPU idle | KV transfer/routing/network -> NIXL/UCX, HCA, cache hit |
| E2E far below single-pool estimate | single-pool test omitted KV cost -> add a KV receiver or discount C_P |
| Regression on new machine | copied NCCL/UCX settings -> redo topology + nccl-tests |
| Cache off but still hits | missing `--no-enable-prefix-caching` |

## Tuning order
Correctness (outputs, MTP acceptance, KV integrity) -> hardware comms -> actual back-end dispatch -> single-pool capacity -> P/D ratio -> scheduling/cache -> production headroom. Change one variable per experiment; keep commands, env vars, logs and raw results.

## Files
- `references/DEPLOYMENT_HANDBOOK.md` - full guide (snapshot; if the user's repo copy is newer, prefer it and update this snapshot).
- `references/BENCHMARK.md` (P/D vs IFB protocol, formulas, reproduction target), `scripts/analyze_benchmark.py`, `templates/bench_ladder.sh`, `templates/start_ifb.sh`.
- `references/validated-images.md` - images that passed acceptance; append after each validation.
- `references/IMAGE_BUILD.md` (how the 1.5.1 image was made), `references/PORTING_GUIDE.md` (port to a new vLLM), `Dockerfile.dual`, `patches/vllm-0.28/` (use `apply_all.py`; `TARGET.json` makes it discoverable).
- `scripts/ngc_info.py`, `auto_build.py`, `probe_base_image.py`, `check_patch_compat.py`, `scaffold_port.py`, `_common.py`, `build_dual_image.sh` (legacy).
- `scripts/find_latest_image.py`, `scripts/verify_deployment.py`, `scripts/compute_pd_ratio.py`.
- `templates/` - `common.env.example`, `start_frontend.sh`, `start_prefill.sh`, `start_decode.sh`.

## Caveats
Numbers (IPs, throughput, concurrency) come from one test environment (8 x RTX 6000D class GPUs over PCIe). Re-measure before quoting them for other hardware. Ratio math is a pool-level model and ignores KV transfer and routing; only end-to-end runs are final.
