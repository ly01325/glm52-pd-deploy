#!/usr/bin/env python3
"""Inspect a Dynamo/vLLM base image (needs docker; no GPU): vLLM version, python, key packages,
which parts of the dual-PCIe feature are already native, and which vLLM internals our modules depend on.

usage: probe_base_image.py <image> [--json]
Verdict lines (KEY=VALUE): VLLM_VERSION, PATCH_SET, FEATURE_STATE (none|partial-native|full-native), NEEDS_*.
"""
import argparse, json, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
from ngc_info import patch_sets, series

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    if not have_docker(): sys.exit("docker not found")
    d = probe_image(a.image); f = d["features"]; ver = d["packages"].get("vllm"); ps = patch_sets().get(series(ver))
    native_ipc = f["envs_has_fi_pcie_ipc"] and f["communicator_has_fi_pcie_ipc"] and f["module_fi_pcie_ipc_exists"]
    native_b12x = f["envs_has_b12x_dma"] and f["communicator_has_b12x_dma"] and f["module_b12x_dma_exists"]
    state = "full-native" if native_ipc and native_b12x else "partial-native" if (native_ipc or native_b12x or f["envs_has_fi_pcie_ipc"] or f["envs_has_b12x_dma"]) else "none"
    d["verdict"] = dict(vllm=ver, patch_set=ps, feature_state=state, native_fi_pcie_ipc=native_ipc, native_b12x_dma=native_b12x,
                        needs_b12x_adapter=not native_b12x, needs_fi_ipc_port=not native_ipc,
                        missing_vllm_api=[k for k, v in d["vllm_api_deps"].items() if not v])
    if a.json: print(json.dumps(d, indent=1)); return
    print(f"image={a.image}\nvllm={ver} python={d['python']} user={d.get('user') or '(default)'} root={d['vllm_root']}")
    print("packages: " + ", ".join(f"{k}={v}" for k, v in d["packages"].items() if v))
    print("features: " + ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in f.items()))
    print("vllm api deps: " + ", ".join(f"{k}={'Y' if v else 'MISSING'}" for k, v in d["vllm_api_deps"].items()))
    v = d["verdict"]
    print(f"\nVLLM_VERSION={ver}\nPATCH_SET={ps or 'none'}\nFEATURE_STATE={state}\nNATIVE_FI_PCIE_IPC={native_ipc}\nNATIVE_B12X_DMA={native_b12x}")
    print(f"MISSING_VLLM_API={','.join(v['missing_vllm_api']) or 'none'}")
    if state != "none": print("NOTE: parts of the feature already exist natively - do NOT blindly run apply_all.py (duplicate edits); follow references/PORTING_GUIDE.md 'native support'.")

if __name__ == "__main__": main()
