#!/usr/bin/env python3
"""Query NVIDIA NGC for official Dynamo vLLM-runtime images WITHOUT pulling them.

The image config blob carries labels/env (ai.vllm.image.tag -> bundled vLLM version, CUDA_VERSION,
DYNAMO_COMMIT_SHA, created), so the Dynamo-tag -> vLLM-version mapping is read from the registry itself.

  ngc_info.py table [--last 6] [--json]     newest stable tags with vLLM/CUDA/created + patch-set status
  ngc_info.py select [--json]               pick build base: newest, and newest that already has a patch set
Anonymous access; no credentials needed for listing. `docker pull` may still need `docker login nvcr.io`
(user `$oauthtoken`, password = NGC API key) if the pull is rejected with 401.
"""
import argparse, json, re, sys, urllib.request
from pathlib import Path

REPO = "nvidia/ai-dynamo/vllm-runtime"
IMAGE = "nvcr.io/" + REPO
SKILL = Path(__file__).resolve().parent.parent
ACCEPT = ",".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                   "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])
STABLE = re.compile(r"^\d+\.\d+\.\d+$")

def vt(s): return tuple(int(x) for x in re.findall(r"\d+", s))

class Ngc:
    def __init__(self):
        with urllib.request.urlopen(f"https://nvcr.io/proxy_auth?scope=repository:{REPO}:pull", timeout=15) as r:
            self.tok = json.load(r)["token"]
    def get(self, path, accept=None):
        req = urllib.request.Request(f"https://nvcr.io/v2/{REPO}/{path}", headers={"Authorization": "Bearer " + self.tok, **({"Accept": accept} if accept else {})})
        with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
    def stable_tags(self):
        return sorted((t for t in self.get("tags/list").get("tags", []) if STABLE.match(t)), key=vt, reverse=True)
    def info(self, tag):
        m = self.get(f"manifests/{tag}", ACCEPT)
        digest = None
        if "manifests" in m:
            amd = next(x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64")
            digest = amd["digest"]; m = self.get(f"manifests/{digest}", ACCEPT)
        cfg = self.get("blobs/" + m["config"]["digest"]); c = cfg.get("config", {})
        lab = c.get("Labels") or {}; env = dict(e.split("=", 1) for e in c.get("Env", []) if "=" in e)
        mv = re.search(r"v(\d+\.\d+\.\d+)", lab.get("ai.vllm.image.tag", "") or env.get("VLLM_IMAGE_TAG", ""))
        return dict(tag=tag, image=f"{IMAGE}:{tag}", amd64_digest=digest, created=cfg.get("created", "")[:10],
                    vllm=mv.group(1) if mv else None, cuda=env.get("CUDA_VERSION"), dynamo_commit=env.get("DYNAMO_COMMIT_SHA"),
                    vllm_build_commit=env.get("VLLM_BUILD_COMMIT"), user=c.get("User") or "")

def patch_sets():
    out = {}
    for t in sorted((SKILL / "patches").glob("vllm-*/TARGET.json")):
        try:
            d = json.loads(t.read_text()); out[d.get("vllm_series") or ".".join(d["vllm"].split(".")[:2])] = t.parent.name
        except Exception: pass
    return out

def series(v): return ".".join(v.split(".")[:2]) if v else None

def collect(n):
    ngc = Ngc(); tags = ngc.stable_tags()[:n]; ps = patch_sets(); rows = []
    for t in tags:
        try: i = ngc.info(t)
        except Exception as e: i = dict(tag=t, image=f"{IMAGE}:{t}", vllm=None, error=str(e)[:60])
        i["patch_set"] = ps.get(series(i.get("vllm")))
        rows.append(i)
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "select"]); ap.add_argument("--last", type=int, default=6); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try: rows = collect(a.last)
    except Exception as e: print(f"ERROR: cannot query NGC ({e}). Offline? Use a known tag with auto_build.py --base-tag.", file=sys.stderr); sys.exit(2)
    newest = rows[0]; patched = next((r for r in rows if r.get("patch_set")), None)
    if a.cmd == "table" and not a.json:
        print(f"{'tag':8} {'vLLM':8} {'CUDA':8} {'created':11} {'dynamo commit':14} patch set")
        for r in rows: print(f"{r['tag']:8} {str(r.get('vllm')):8} {str(r.get('cuda')):8} {r.get('created',''):11} {str(r.get('dynamo_commit') or '')[:12]:14} {r.get('patch_set') or '-  (needs porting)'}")
        return
    res = dict(newest=newest, newest_with_patch_set=patched,
               needs_port=bool(newest and not newest.get("patch_set")),
               recommendation=("build on newest" if newest.get("patch_set") else "port patch to vLLM %s (scripts/scaffold_port.py), or fall back to %s" % (newest.get("vllm"), patched["tag"] if patched else "none")))
    if a.json or a.cmd == "select" and a.json: print(json.dumps(res, indent=1)); return
    print(f"NEWEST_OFFICIAL={newest['image']}  (vLLM {newest.get('vllm')}, CUDA {newest.get('cuda')}, created {newest.get('created')})")
    print(f"NEWEST_PATCH_SET={newest.get('patch_set') or 'none'}")
    if patched: print(f"NEWEST_PATCHED_BASE={patched['image']}  (vLLM {patched.get('vllm')}, patch set {patched['patch_set']})")
    print(f"NEEDS_PORT={'yes' if res['needs_port'] else 'no'}")
    print("RECOMMENDATION=" + res["recommendation"])

if __name__ == "__main__": main()
