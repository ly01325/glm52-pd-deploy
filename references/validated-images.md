# Validated runtime images

Only images listed here have passed the full P/D acceptance (see SKILL.md "Image policy").
`scripts/find_latest_image.py` parses the `VALIDATED:` lines. Append one line per newly validated image.

Format: `VALIDATED: <image:tag> | <key facts>`

VALIDATED: dynamo-vllm-runtime:1.5.1-vllm028-pcie-dual | Dynamo 1.5.1 (commit 2093662a) on vllm/vllm-openai:v0.28.0-ubuntu2404 (commit 2cf0a691) with B12X PCIe DMA + FlashInfer PCIe IPC AllReduce | validated 2026-10-11 | 1P1D P=B12X D=FlashInfer IPC + PIECEWISE | MTP acceptance 95.64% (IFB 94.54%) | Gate A C10 0.586 req/s, Gate B C12 0.647 req/s | model GLM-5.2-NVFP4

Base image lineage (per the image builder, following the NVIDIA Dynamo docs): official `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1` (Dynamo v1.5.1 pins vLLM v0.28.0). Build recipe: `references/IMAGE_BUILD.md` (patched running container + `docker commit`, image ID 00bceaa0c57f..., ~45.3 GB, 2026-10-09; NOT Dockerfile-reproducible). Pip: b12x 1.3.0, flashinfer-python 0.7.0.post1.

## Required evidence for a new entry
1. `verify_deployment.py` passes for the P and D logs.
2. N=100, 1024-output run at C1: 100% completion, MTP acceptance >= IFB reference - 1 pp.
3. Gate A/B boundary points re-run (N=100); record req/s in the line above.
4. No Traceback / CUDA error / OOM in either log.
