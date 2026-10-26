#!/usr/bin/env python3
"""Check whether a patch set applies to an image's vLLM, WITHOUT running the image (docker create + cp).

usage: check_patch_compat.py <image> [--patch-set vllm-0.28] [--vllm-root /usr/local/lib/python3.12/dist-packages/vllm]
       check_patch_compat.py --from-dir <dir containing vllm/> [--patch-set vllm-0.28]      (offline / tests)
Exit 0 = COMPATIBLE, 1 = INCOMPATIBLE (lists every anchor that needs porting), 2 = cannot check.
"""
import argparse, subprocess, sys, tempfile
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
from ngc_info import patch_sets, series

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image", nargs="?"); ap.add_argument("--patch-set"); ap.add_argument("--vllm-root")
    ap.add_argument("--from-dir"); a = ap.parse_args(); tmp = tempfile.mkdtemp(prefix="glm52-compat-")
    if a.from_dir: tree = Path(a.from_dir) / "vllm"; ps = a.patch_set
    else:
        if not (a.image and have_docker()): sys.exit("need an image and docker (or --from-dir)")
        pr = probe_image(a.image); ver = pr["packages"].get("vllm"); ps = a.patch_set or patch_sets().get(series(ver))
        print(f"image vLLM {ver}; patch set: {ps or 'NONE'}")
        if not ps: print("RESULT: NO PATCH SET for this vLLM series -> scripts/scaffold_port.py"); sys.exit(1)
        tree = extract_vllm_files(a.image, a.vllm_root or pr["vllm_root"], tmp)
    if not ps: sys.exit("--patch-set required with --from-dir")
    pdir = SKILL / "patches" / ps
    r = subprocess.run([sys.executable, str(pdir / "apply_all.py"), "--root", str(tree), "--check", "--new-files-dir", str(pdir / "final")], text=True, capture_output=True)
    print(r.stdout.split("== compatibility check ==")[-1].strip() if "== compatibility check ==" in r.stdout else (r.stdout + r.stderr)[-1500:])
    sys.exit(r.returncode if r.returncode in (0, 1) else 2)

if __name__ == "__main__": main()
