"""Shared helpers for the image build scripts (stdlib only)."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
# files the dual-PCIe patch touches, relative to the vllm package root
PATCH_FILES = {
    "envs.py": "envs.py",
    "cuda_communicator.py": "distributed/device_communicators/cuda_communicator.py",
    "parallel_state.py": "distributed/parallel_state.py",
    "kernel_warmup.py": "model_executor/warmup/kernel_warmup.py",
    "flashinfer_mla_sparse_sm120.py": "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
}
NEW_MODULES = ("b12x_pcie_dma.py", "flashinfer_pcie_ipc_all_reduce.py")

def have_docker(): return shutil.which("docker") is not None

def run(cmd, check=True, capture=True, timeout=None, echo=False):
    if echo: print("+ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{(r.stderr or '')[-1500:]}")
    return r

PROBE_CODE = r'''
import importlib.metadata as md, importlib.util, json, os, sys, glob
out = {"python": sys.version.split()[0], "python_exe": sys.executable}
spec = importlib.util.find_spec("vllm")
root = list(spec.submodule_search_locations)[0] if spec else None
out["vllm_root"] = root
def rd(rel):
    try: return open(os.path.join(root, rel), errors="replace").read()
    except Exception: return ""
pk = {}
for n in ["vllm","flashinfer-python","flashinfer-cubin","flashinfer-jit-cache","b12x","nccl-extensions","nvidia-cudnn-frontend","torch","nixl","ai-dynamo","ai-dynamo-runtime"]:
    try: pk[n] = md.version(n)
    except Exception: pk[n] = None
out["packages"] = pk
fi = importlib.util.find_spec("flashinfer"); ws = False
if fi:
    base = list(fi.submodule_search_locations)[0]
    for f in glob.glob(base + "/comm/**/*.py", recursive=True):
        try:
            if "PcieIpcAllReduceWorkspace" in open(f, errors="replace").read(): ws = True; break
        except Exception: pass
cfg = "".join(open(f, errors="replace").read() for f in glob.glob(os.path.join(root, "config", "*.py"))) if root else ""
out["features"] = {
  "envs_has_fi_pcie_ipc": "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC" in rd("envs.py"),
  "envs_has_b12x_dma": "VLLM_ALLREDUCE_USE_B12X_PCIE_DMA" in rd("envs.py"),
  "communicator_has_fi_pcie_ipc": "FlashInferPcieIpcAllReduce" in rd("distributed/device_communicators/cuda_communicator.py"),
  "communicator_has_b12x_dma": "B12xPcieDmaAllReduce" in rd("distributed/device_communicators/cuda_communicator.py"),
  "module_fi_pcie_ipc_exists": os.path.exists(os.path.join(root or "", "distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py")),
  "module_b12x_dma_exists": os.path.exists(os.path.join(root or "", "distributed/device_communicators/b12x_pcie_dma.py")),
  "flashinfer_has_PcieIpcAllReduceWorkspace": ws,
  "sm120_sparse_mla_file_exists": os.path.exists(os.path.join(root or "", "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py")),
  "sm120_has_supports_dense_mha_prefill": "supports_dense_mha_prefill" in rd("v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"),
}
out["vllm_api_deps"] = {   # internals the two new modules rely on
  "parallel_state.in_the_same_node_as": "def in_the_same_node_as" in rd("distributed/parallel_state.py"),
  "warmup/flashinfer_autotune_cache.resolve_flashinfer_autotune_file": "def resolve_flashinfer_autotune_file" in rd("model_executor/warmup/flashinfer_autotune_cache.py"),
  "config.parallel_config.use_ubatching": "use_ubatching" in cfg,
  "kernel_warmup.minimax_m3_msa_warmup_anchor": "minimax_m3_msa_warmup(worker)" in rd("model_executor/warmup/kernel_warmup.py"),
  "logger.info_once": "info_once" in rd("logger.py"),
}
print("PROBE_JSON=" + json.dumps(out))
'''

def probe_image(image):
    """Run read-only inspection inside the image (no GPU needed). Returns dict."""
    r = run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", PROBE_CODE], check=False, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON=")), None)
    if not line: raise RuntimeError("probe failed: " + (r.stderr or r.stdout)[-800:])
    d = json.loads(line[len("PROBE_JSON="):])
    ins = run(["docker", "image", "inspect", image, "--format", "{{json .Config}}"], check=False)
    try:
        cfg = json.loads(ins.stdout); d["user"] = cfg.get("User") or ""; d["image_env"] = [e for e in cfg.get("Env", []) if e.split("=")[0] in ("DYNAMO_COMMIT_SHA", "VLLM_IMAGE_TAG", "CUDA_VERSION")]
    except Exception: d["user"] = ""
    return d

def extract_vllm_files(image, vllm_root, dest):
    """Copy the files the patch touches out of an image (docker create + cp; nothing is started). dest/vllm/<rel>"""
    cid = run(["docker", "create", image]).stdout.strip()
    try:
        for rel in PATCH_FILES.values():
            tgt = Path(dest) / "vllm" / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
            run(["docker", "cp", f"{cid}:{vllm_root}/{rel}", str(tgt)], check=False)
    finally:
        run(["docker", "rm", "-f", cid], check=False)
    return Path(dest) / "vllm"

def series_digits(vllm_version):  # "0.28.0" -> "028", "0.31.0" -> "031"
    p = vllm_version.split("."); return f"{int(p[0])}{int(p[1]):02d}"
