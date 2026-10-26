#!/usr/bin/env python3
"""Pull the newest official Dynamo vLLM-runtime image, find the vLLM it ships, and try to build the dual-PCIe image on it.

  auto_build.py plan [--base-tag 1.6.0] [--policy newest|newest-patched]     what would happen (needs only network)
  auto_build.py run  [--base-tag ...] [--policy ...] [--method dockerfile|commit] [--no-pull] [--dry-run]

Policy:  newest          -> build on the newest official Dynamo tag; if no patch set exists for its vLLM series, create a
                            scaffold (scripts/scaffold_port.py) and STOP (exit 3) so an agent can port, then re-run.
         newest-patched  -> newest tag whose vLLM series already has a patch set (e.g. 1.5.1 / vLLM 0.28).
Exit codes: 0 built+smoke OK | 2 environment problem | 3 patch port needed / incompatible anchors | 4 build or smoke failed.
A built image is NOT validated. Run the GPU acceptance in SKILL.md before recording it in references/validated-images.md.
"""
import argparse, json, os, subprocess, sys, datetime
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
from ngc_info import Ngc, patch_sets, series, STABLE, vt, IMAGE

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run"]); ap.add_argument("--base-tag"); ap.add_argument("--policy", choices=["newest", "newest-patched"], default="newest")
    ap.add_argument("--method", choices=["dockerfile", "commit"], default="dockerfile"); ap.add_argument("--repo", default="dynamo-vllm-runtime")
    ap.add_argument("--no-pull", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--last", type=int, default=6)
    ap.add_argument("--report-dir", default=str(SKILL / "build-reports")); a = ap.parse_args()
    dry = a.dry_run or a.cmd == "plan"

    # 1. choose the base from the registry (labels give the bundled vLLM; nothing is pulled yet)
    try:
        ngc = Ngc(); tags = ngc.stable_tags()[:a.last]
    except Exception as e: print(f"ERROR cannot reach NGC: {e}\nPass --base-tag with a known tag and run on a networked host.", file=sys.stderr); sys.exit(2)
    ps = patch_sets()
    def row(t):
        r = ngc.info(t); r["patch_set"] = ps.get(series(r.get("vllm"))); return r
    all_rows = [row(t) for t in tags]                      # newest first
    if a.base_tag: rows = [row(a.base_tag)]
    elif a.policy == "newest-patched": rows = [r for r in all_rows if r["patch_set"]] or sys.exit("no recent tag has a matching patch set")
    else: rows = all_rows
    base = rows[0]
    print(f"BASE={base['image']}  vLLM {base['vllm']}  CUDA {base.get('cuda')}  created {base.get('created')}  patch set: {base['patch_set'] or 'NONE'}")
    if not base["patch_set"]:
        print(f"-> no patch set for vLLM {series(base['vllm'])}: the dual-PCIe patch must be PORTED (references/PORTING_GUIDE.md).")
        if a.cmd == "run" and have_docker() and not a.dry_run:
            if not a.no_pull: run(["docker", "pull", base["image"]], capture=False, echo=True)
            run([sys.executable, str(SKILL / "scripts" / "scaffold_port.py"), "--image", base["image"]], capture=False, echo=True)
        else: print(f"   next: python3 scripts/scaffold_port.py --image {base['image']}")
        nb = next((r for r in all_rows if r['patch_set']), None)
        if nb: print(f"   fallback with an existing patch set: --base-tag {nb['tag']}  (vLLM {nb['vllm']})")
        sys.exit(3)
    pdir = SKILL / "patches" / base["patch_set"]; tgt = json.loads((pdir / "TARGET.json").read_text())
    if tgt.get("status", "").startswith("SCAFFOLD"): print(f"patch set {base['patch_set']} is still a SCAFFOLD (not ported). Finish PORTING_TODO.md first."); sys.exit(3)
    tag = f"{base['tag']}-vllm{series_digits(base['vllm'])}-pcie-dual"; target = f"{a.repo}:{tag}"
    pip = tgt.get("pip", {})
    print(f"TARGET={target}  patch set={base['patch_set']}  method={a.method}")
    steps = [f"docker pull {base['image']}", f"probe_base_image.py {base['image']}  (vLLM/packages/native-feature check)",
             f"check_patch_compat.py {base['image']} --patch-set {base['patch_set']}  (anchor check, no container run)",
             f"build ({a.method}) -> {target}", f"smoke: import check inside {target}", f"write build report in {a.report_dir}"]
    print("STEPS:\n  " + "\n  ".join(steps))
    if dry:
        print("(plan only - nothing executed)"); return
    if not have_docker(): print("ERROR docker not found", file=sys.stderr); sys.exit(2)

    # 2. pull
    if not a.no_pull:
        r = run(["docker", "pull", base["image"]], check=False, capture=False, echo=True)
        if r.returncode: print("pull failed. If 401: `docker login nvcr.io` (user $oauthtoken, password = NGC API key).", file=sys.stderr); sys.exit(2)
    # 3. probe + compat
    pr = probe_image(base["image"]); v = pr["features"]; ver = pr["packages"]["vllm"]
    print(f"probe: vllm={ver} python={pr['python']} user={pr.get('user') or '(default)'}")
    if ver != tgt["vllm"] and series(ver) != tgt["vllm_series"]: print(f"vLLM {ver} does not match patch set {tgt['vllm']}"); sys.exit(3)
    if any((v["envs_has_fi_pcie_ipc"], v["envs_has_b12x_dma"], v["module_fi_pcie_ipc_exists"], v["module_b12x_dma_exists"])):
        print("NOTE: the base already has some of the feature natively; apply_all.py would duplicate it. Follow PORTING_GUIDE 'native support'."); sys.exit(3)
    cc = run([sys.executable, str(SKILL / "scripts" / "check_patch_compat.py"), base["image"], "--patch-set", base["patch_set"]], check=False)
    print(cc.stdout); 
    if cc.returncode: print("patch does not apply cleanly -> port it (PORTING_GUIDE.md)"); sys.exit(3)
    # 4. build
    have = pr["packages"]; skip_fi = "0"
    if have.get("flashinfer-python") and vt(have["flashinfer-python"]) > vt(pip.get("flashinfer-python", "0")): skip_fi = "1"; print(f"keeping base flashinfer-python {have['flashinfer-python']} (newer than pinned {pip.get('flashinfer-python')})")
    ba = ["--build-arg", f"BASE={base['image']}", "--build-arg", f"PATCH_SET={base['patch_set']}", "--build-arg", f"RUN_USER={pr.get('user') or ''}", "--build-arg", f"SKIP_FLASHINFER={skip_fi}",
          "--build-arg", f"B12X_VER={pip.get('b12x', '1.3.0')}", "--build-arg", f"FI_VER={pip.get('flashinfer-python', '0.7.0.post1')}", "--build-arg", f"FI_CUBIN_VER={pip.get('flashinfer-cubin', pip.get('flashinfer-python', '0.7.0.post1'))}",
          "--build-arg", f"NCCL_EXT_VER={pip.get('nccl-extensions', '0.1.0')}", "--build-arg", f"CUDNN_FE_VER={pip.get('nvidia-cudnn-frontend', '1.30.0')}"]
    if a.method == "dockerfile":
        r = run(["docker", "build", "-f", str(SKILL / "Dockerfile.dual"), *ba, "-t", target, str(SKILL)], check=False, capture=False, echo=True)
    else:
        env = dict(os.environ, BASE=base["image"], TARGET=target, PATCH_DIR=str(pdir)); r = subprocess.run(["bash", str(SKILL / "scripts" / "build_dual_image.sh")], env=env)
    if r.returncode: print("BUILD FAILED. Read the last error above; typical causes: pip wheel unavailable for this CUDA, anchor mismatch, no network.", file=sys.stderr); sys.exit(4)
    # 5. smoke
    sm = run(["docker", "run", "--rm", "--entrypoint", "python3", target, "-c",
              "import vllm, vllm.envs as e, vllm.distributed.device_communicators.b12x_pcie_dma as b, vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce as f; "
              "print('smoke ok', vllm.__version__, e.VLLM_ALLREDUCE_USE_B12X_PCIE_DMA, e.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC)"], check=False)
    ok = sm.returncode == 0 and "smoke ok" in sm.stdout; print(sm.stdout.strip() or sm.stderr[-600:])
    # 6. report
    Path(a.report_dir).mkdir(parents=True, exist_ok=True); rep = Path(a.report_dir) / f"{tag}.md"
    dig = run(["docker", "image", "inspect", target, "--format", "{{.Id}}"], check=False).stdout.strip()
    rep.write_text(f"""# Build report {target}
- date: {datetime.datetime.now().isoformat(timespec='seconds')}
- base: {base['image']} (amd64 digest {base.get('amd64_digest')}), vLLM {ver}, CUDA {base.get('cuda')}, Dynamo commit {base.get('dynamo_commit')}
- patch set: {base['patch_set']} ({tgt.get('validated_on')})
- method: {a.method}; pip pins: {pip}; kept base flashinfer: {skip_fi == '1'}
- image id: {dig}
- smoke import test: {'PASS' if ok else 'FAIL'}
- STATUS: BUILT, NOT VALIDATED. Next: GPU acceptance (SKILL.md steps 5-7), verify_deployment.py on real logs, then add a VALIDATED line.
""")
    print(f"report: {rep}\nIMAGE_BUILT={target}\nSTATUS={'BUILT_NOT_VALIDATED' if ok else 'SMOKE_FAILED'}")
    sys.exit(0 if ok else 4)

if __name__ == "__main__": main()
