#!/usr/bin/env python3
"""Create patches/vllm-<series>/ as a starting point for porting the dual-PCIe patch to a new vLLM series.

usage: scaffold_port.py --image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0 [--base-set vllm-0.28]   (needs docker)
       scaffold_port.py --from-dir <dir with vllm/> --vllm 0.31.0 [--python 3.12] [--base-set vllm-0.28]    (offline)
Creates original/ (pristine files of the NEW vLLM), final/ (copy of original/ + the two back-end modules from the base set,
to be edited), apply_all.py (copied from the base set; its anchors are what you must fix), TARGET.json, DRIFT.md
(per-file drift vs the base set's originals + the --check anchor report) and PORTING_TODO.md.
Porting itself needs judgement: follow references/PORTING_GUIDE.md.
"""
import argparse, difflib, json, shutil, subprocess, sys, tempfile
sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--from-dir"); ap.add_argument("--vllm"); ap.add_argument("--python", default="3.12")
    ap.add_argument("--base-set", default="vllm-0.28"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    base = SKILL / "patches" / a.base_set
    if not (base / "apply_all.py").exists(): sys.exit(f"base set missing: {base}")
    pip = {}; root_in_image = "/usr/local/lib/python3.12/dist-packages/vllm"
    if a.from_dir:
        tree = Path(a.from_dir) / "vllm"; ver = a.vllm or sys.exit("--vllm required with --from-dir")
    else:
        if not (a.image and have_docker()): sys.exit("need --image and docker (or --from-dir)")
        pr = probe_image(a.image); ver = pr["packages"]["vllm"]; root_in_image = pr["vllm_root"]; a.python = pr["python"].rsplit(".", 1)[0]
        pip = {k: v for k, v in pr["packages"].items() if v and k != "vllm"}
        tree = extract_vllm_files(a.image, root_in_image, tempfile.mkdtemp(prefix="glm52-scaffold-"))
    series = ".".join(ver.split(".")[:2]); dest = SKILL / "patches" / f"vllm-{series}"
    if dest.exists() and not a.force: sys.exit(f"{dest} exists (use --force to overwrite)")
    for sub in ("original", "final"): (dest / sub).mkdir(parents=True, exist_ok=True)
    missing = []
    for name, rel in PATCH_FILES.items():
        src = tree / rel
        if not src.exists(): missing.append(rel); continue
        shutil.copy2(src, dest / "original" / name); shutil.copy2(src, dest / "final" / name)
    for m in NEW_MODULES: shutil.copy2(base / "final" / m, dest / "final" / m)
    shutil.copy2(base / "apply_all.py", dest / "apply_all.py")
    tgt = json.loads((base / "TARGET.json").read_text())
    tgt.update(vllm=ver, vllm_series=series, python=a.python, vllm_site_packages_root=root_in_image, validated_on=None, status="SCAFFOLD - not ported, not validated",
               pip_in_base_image=pip, ported_from=a.base_set)
    (dest / "TARGET.json").write_text(json.dumps(tgt, indent=2) + "\n")
    # drift report
    lines = [f"# Drift: {a.base_set} originals -> vLLM {ver} originals\n"]
    for name in PATCH_FILES:
        o, n = base / "original" / name, dest / "original" / name
        if not o.exists() or not n.exists(): lines.append(f"- `{name}`: MISSING in {'new' if o.exists() else 'base'} set\n"); continue
        d = list(difflib.unified_diff(o.read_text().splitlines(), n.read_text().splitlines(), lineterm="", n=2))
        adds = sum(1 for l in d if l.startswith("+") and not l.startswith("+++")); dels = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        lines.append(f"## {name}: +{adds} / -{dels} lines" + ("  (identical)" if not d else ""))
        if d and len(d) < 400: lines.append("```diff\n" + "\n".join(d) + "\n```")
        elif d: lines.append(f"(diff too large to inline; run: diff -u {o} {n})")
        lines.append("")
    chk = subprocess.run([sys.executable, str(dest / "apply_all.py"), "--root", str(tree), "--check", "--new-files-dir", str(dest / "final")], capture_output=True, text=True)
    out = chk.stdout.split("== compatibility check ==")[-1].strip()
    lines += ["## Anchor check of the copied apply_all.py against the new vLLM", "```", out or chk.stderr[-800:], "```"]
    (dest / "DRIFT.md").write_text("\n".join(lines) + "\n")
    (dest / "PORTING_TODO.md").write_text(f"""# Port dual-PCIe patch to vLLM {ver}
Status: scaffold only. Read references/PORTING_GUIDE.md first, then DRIFT.md here.
1. Run `python3 scripts/probe_base_image.py <image>` - which parts are already native?
2. Edit `final/*.py` (start: copies of the new originals) so they contain the intended behaviour (see PORTING_GUIDE "Intent of each edit").
3. Make `apply_all.py` reproduce `final/` from `original/`: fix each MISS anchor in DRIFT.md until
   `python3 apply_all.py --root <tree with original/> --check` says COMPATIBLE and a real apply is byte-identical to final/.
4. `python3 -m py_compile final/*.py`; `python3 scripts/auto_build.py run --base-tag <dynamo tag>` to build and smoke-test.
5. Runtime validation needs GPUs (SKILL.md steps 5-7). Only then add a VALIDATED line and set TARGET.json status.
Missing source files in the new image: {missing or 'none'}
""")
    print(f"scaffold created: {dest}\n  -> read {dest}/DRIFT.md and {dest}/PORTING_TODO.md\n" + (out.splitlines()[-1] if out else ""))

if __name__ == "__main__": main()
