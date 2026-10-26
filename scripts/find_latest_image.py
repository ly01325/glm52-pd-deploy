#!/usr/bin/env python3
"""Find the newest GLM-5.2 P/D runtime image and say whether it is validated.

Image naming convention:  <repo>:<dynamo-ver>-vllm<NNN>-<suffix>
  e.g. dynamo-vllm-runtime:1.5.1-vllm028-pcie-dual

Sources (all optional, merged):
  1. local docker images                        (needs `docker`)
  2. GLM52_TAGS_CMD  - any shell command printing one tag per line
  3. GLM52_REGISTRY_URL (+ GLM52_REGISTRY_REPO) - Docker Registry v2 /tags/list
     auth: GLM52_REGISTRY_TOKEN (bearer) or GLM52_REGISTRY_USER/GLM52_REGISTRY_PASS
  4. GLM52_SKOPEO_REF - e.g. registry.example.com/team/dynamo-vllm-runtime (needs skopeo)
Official base image (unless --no-official): NVIDIA NGC  nvcr.io/nvidia/ai-dynamo/vllm-runtime:<dynamo-ver>
  (anonymous token + registry v2 tag list). Our image is built on top of it, so a newer official tag
  means the base moved and a rebuild/re-validation of the patched image is due.
Upstream check (unless --no-upstream): latest vLLM release on GitHub (informational).

Output: human summary, then KEY=VALUE lines for agents.  Exit 2 if no candidate image found.
"""
import argparse, json, os, re, shutil, subprocess, sys, urllib.request, base64
from pathlib import Path

TAG_RE = re.compile(r"^(\d+(?:\.\d+)*)-vllm(\d+)-(.+)$")
HERE = Path(__file__).resolve().parent
VALIDATED_FILE = HERE.parent / "references" / "validated-images.md"

def ver_tuple(s):
    return tuple(int(x) for x in re.findall(r"\d+", s))

def vllm_digits_to_version(d):
    # 028 -> 0.28.0 ; 0251 -> 0.25.1 ; 31 -> 0.31
    if len(d) == 3: return (0, int(d[1:]), 0)
    if len(d) == 4: return (0, int(d[1:3]), int(d[3:]))
    return (0, int(d), 0)

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception as e:
        return 1, ""

def local_tags(repo):
    if not shutil.which("docker"): return []
    rc, out = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if rc: return []
    return [l.split(":", 1)[1] for l in out.splitlines() if l.startswith(repo + ":")]

def registry_tags():
    cmd = os.environ.get("GLM52_TAGS_CMD")
    if cmd:
        rc, out = run(cmd, 60); return [l.strip() for l in out.splitlines() if l.strip()], "GLM52_TAGS_CMD"
    url, rrepo = os.environ.get("GLM52_REGISTRY_URL"), os.environ.get("GLM52_REGISTRY_REPO")
    if url and rrepo:
        req = urllib.request.Request(f"{url.rstrip('/')}/v2/{rrepo}/tags/list")
        tok = os.environ.get("GLM52_REGISTRY_TOKEN")
        if tok: req.add_header("Authorization", f"Bearer {tok}")
        elif os.environ.get("GLM52_REGISTRY_USER"):
            b = base64.b64encode(f"{os.environ['GLM52_REGISTRY_USER']}:{os.environ.get('GLM52_REGISTRY_PASS','')}".encode()).decode()
            req.add_header("Authorization", f"Basic {b}")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r).get("tags") or [], "registry v2"
        except Exception as e:
            print(f"WARN: registry query failed: {e}", file=sys.stderr); return [], "registry v2 (failed)"
    ref = os.environ.get("GLM52_SKOPEO_REF")
    if ref and shutil.which("skopeo"):
        rc, out = run(["skopeo", "list-tags", f"docker://{ref}"], 60)
        try: return json.loads(out).get("Tags", []), "skopeo"
        except Exception: return [], "skopeo (failed)"
    return [], None

def parse(tags, suffix):
    res = []
    for t in dict.fromkeys(tags):
        m = TAG_RE.match(t)
        if not m or m.group(3) != suffix: continue
        res.append(dict(tag=t, dynamo=ver_tuple(m.group(1)), vllm=vllm_digits_to_version(m.group(2)),
                        dynamo_s=m.group(1), vllm_s=".".join(map(str, vllm_digits_to_version(m.group(2))))))
    res.sort(key=lambda x: (x["vllm"], x["dynamo"]), reverse=True)
    return res

def validated():
    if not VALIDATED_FILE.exists(): return {}
    out = {}
    for l in VALIDATED_FILE.read_text().splitlines():
        if l.startswith("VALIDATED:"):
            img, _, facts = l[len("VALIDATED:"):].strip().partition("|")
            out[img.strip()] = facts.strip()
    return out

NGC_REPO = "nvidia/ai-dynamo/vllm-runtime"
NGC_IMAGE = "nvcr.io/" + NGC_REPO
STABLE_RE = re.compile(r"^\d+\.\d+\.\d+$")

def ngc_official_tags():
    """Anonymous pull token + /tags/list on nvcr.io. Returns (stable_tags_sorted, all_tags) or (None, None)."""
    try:
        with urllib.request.urlopen(f"https://nvcr.io/proxy_auth?scope=repository:{NGC_REPO}:pull", timeout=10) as r:
            tok = json.load(r)["token"]
        req = urllib.request.Request(f"https://nvcr.io/v2/{NGC_REPO}/tags/list", headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            tags = json.load(r).get("tags", [])
        return sorted((t for t in tags if STABLE_RE.match(t)), key=ver_tuple), tags
    except Exception as e:
        print(f"WARN: NGC lookup failed: {e}", file=sys.stderr); return None, None

def upstream(repo):
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/latest", headers={"User-Agent": "glm52-skill"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.load(r).get("tag_name")
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("GLM52_IMAGE_REPO", "dynamo-vllm-runtime"))
    ap.add_argument("--suffix", default=os.environ.get("GLM52_IMAGE_SUFFIX", "pcie-dual"))
    ap.add_argument("--no-upstream", action="store_true")
    ap.add_argument("--no-official", action="store_true", help="skip the NGC official-image lookup")
    a = ap.parse_args()

    loc = local_tags(a.repo); reg, regsrc = registry_tags()
    cands = parse(loc + reg, a.suffix)
    ok = validated()
    print(f"repo={a.repo} suffix={a.suffix}")
    print(f"local tags: {len(loc)}; registry tags: {len(reg)} ({regsrc or 'registry not configured'})")
    if not cands:
        print("NO candidate image found. Configure GLM52_REGISTRY_URL+GLM52_REGISTRY_REPO, GLM52_TAGS_CMD, or GLM52_SKOPEO_REF, or build the image.")
        print("FALLBACK_VALIDATED_IMAGE=" + (next(iter(ok)) if ok else ""))
        if not a.no_official:
            stable, _ = ngc_official_tags()
            if stable: print(f"OFFICIAL_LATEST={NGC_IMAGE}:{stable[-1]}   (official base only; B12X + FlashInfer PCIe-IPC patched image must be built/pulled separately)")
        sys.exit(2)
    best = cands[0]; image = f"{a.repo}:{best['tag']}"
    is_val = image in ok
    where = "+".join(x for x, y in (("local", best['tag'] in loc), ("registry", best['tag'] in reg)) if y)
    print(f"\nNewest image : {image}  (vLLM {best['vllm_s']}, Dynamo {best['dynamo_s']}; found in: {where})")
    print(f"Validated    : {'YES - ' + ok[image] if is_val else 'NO - run full acceptance before production (see SKILL.md)'}")
    newest_val = next((c for c in cands if f"{a.repo}:{c['tag']}" in ok), None)
    if not is_val and newest_val:
        print(f"Newest validated: {a.repo}:{newest_val['tag']}  <- use this unless you are validating the newer one")
    if len(cands) > 1:
        print("Other candidates: " + ", ".join(c['tag'] for c in cands[1:6]))
    notes = []
    off_latest = off_base_ok = ""
    if not a.no_official:
        stable, alltags = ngc_official_tags()
        if stable:
            off_latest = stable[-1]; base = best["dynamo_s"]
            off_base_ok = "yes" if base in stable else "no"
            print(f"Official NGC : {NGC_IMAGE}  latest stable tag {off_latest}; base of this image ({base}) {'exists' if off_base_ok == 'yes' else 'NOT found'} in NGC")
            if ver_tuple(off_latest) > best["dynamo"]:
                print(f"               -> NGC has a newer Dynamo image than this one's base ({off_latest} > {base}). Rebuild the patched image on it and re-validate; do not swap silently.")
                print("               -> next: python3 scripts/auto_build.py plan   (then `run`; see SKILL.md step 0)")
                notes.append(f"Dynamo:{off_latest}")
        else:
            print("Official NGC : lookup failed or offline")
    if not a.no_upstream:
        tag = upstream("vllm-project/vllm")
        if tag: print(f"Upstream vLLM: latest GitHub release {tag} (informational; Dynamo images pin their own vLLM, e.g. Dynamo 1.5.x -> vLLM 0.28.0)")
        else: print("Upstream vLLM: (lookup failed or offline)")
    print(f"\nLATEST_IMAGE={image}")
    print(f"LATEST_VALIDATED={'yes' if is_val else 'no'}")
    print(f"RECOMMENDED_IMAGE={image if is_val else (a.repo + ':' + newest_val['tag'] if newest_val else image)}")
    print(f"UPSTREAM_NEWER={','.join(notes) if notes else 'none'}")
    print(f"OFFICIAL_LATEST={NGC_IMAGE + ':' + off_latest if off_latest else ''}")
    print(f"OFFICIAL_BASE_OF_RECOMMENDED={NGC_IMAGE + ':' + best['dynamo_s'] if off_base_ok == 'yes' else ''}")

if __name__ == "__main__":
    main()
