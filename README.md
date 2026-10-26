# glm52-pd-deploy

An **agent skill** (works with Claude Code and other agents that read `SKILL.md`) that teaches an agent how to deploy
**GLM-5.2-NVFP4 prefill/decode (P/D) disaggregated serving** on vLLM + NVIDIA Dynamo, size the P:D ratio, verify the
deployment from logs, and **find the newest official Dynamo image and try to build the dual-PCIe image on it**.

Validated stack (single test environment, 8 x PCIe GPUs per node, TP8 + EP, MTP3, FP8 KV): Dynamo 1.5.1 / vLLM 0.28.0,
Prefill = B12X PCIe DMA all-reduce, Decode = FlashInfer PCIe IPC all-reduce + CUDA Graph `PIECEWISE`, KV over NIXL/UCX.
In that environment 1P1D reached about 3.0x (Gate A) / 3.36x (Gate B) the throughput of a single-node IFB NCCL baseline.
**Re-measure on your own hardware before relying on any number.**

## Install

Clone into your agent's skills directory (Claude Code example):

```bash
git clone https://github.com/ly01325/glm52-pd-deploy.git ~/.claude/skills/glm52-pd-deploy
```

or use it per project (`.claude/skills/glm52-pd-deploy`) or point any agent at `SKILL.md`. Then ask the agent, e.g.
"deploy GLM-5.2 with P/D disaggregation", "find the latest Dynamo image and build the runtime image", or
"verify this P/D deployment from these logs".

Requirements for the scripts: Python 3.9+ (stdlib only); `docker` for building/probing; network access to `nvcr.io` for the
image lookup (listing is anonymous; `docker pull` may need `docker login nvcr.io`).

## What is inside

| Path | Purpose |
|---|---|
| `SKILL.md` | Entry point: hard rules, the 8-step workflow, image decision tree, troubleshooting |
| `scripts/ngc_info.py` | Dynamo tag -> bundled vLLM / CUDA, read from NGC image labels (no pull) |
| `scripts/auto_build.py` | Pull newest official image, probe, check patch, build (`Dockerfile.dual`), smoke test, report |
| `scripts/check_patch_compat.py`, `probe_base_image.py`, `scaffold_port.py` | Anchor check without running the image; base-image inspection; port scaffolding for a new vLLM |
| `scripts/verify_deployment.py`, `compute_pd_ratio.py`, `find_latest_image.py` | Log verification; ratio math; derived-image discovery |
| `patches/vllm-0.28/` | The dual-PCIe patch for vLLM 0.28.0 (`apply_all.py`, `original/`, `final/`, `TARGET.json`) |
| `references/` | Deployment handbook, image build notes, **porting guide**, validated-image list |
| `templates/` | Frontend / Prefill / Decode launch scripts and an env example |

## Status and limits (please read)

- The patch only targets **vLLM 0.28.0**. The newest official Dynamo (1.6.0) ships vLLM 0.31.0, so `auto_build.py` will stop and ask for a port
  (`references/PORTING_GUIDE.md`); it does not port automatically.
- The build pipeline (`Dockerfile.dual`, `auto_build.py`) has been exercised only against a mocked `docker`, not on a real build host with GPUs.
  Pip wheel availability for your CUDA version is unverified.
- A built image is **not validated**. Run the GPU acceptance gates in `SKILL.md` before using it.
- The patch modifies vLLM internals; use at your own risk and keep it off production until validated.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Files under `patches/vllm-0.28/{original,final}` derive from vLLM (Apache-2.0).
