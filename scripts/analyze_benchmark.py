#!/usr/bin/env python3
"""Analyze `vllm bench serve` result JSONs for an IFB (aggregate) deployment and a P/D deployment, apply the SLO gates,
and compute the P/D gain over IFB.  Stdlib only (no numpy).

usage:
  analyze_benchmark.py --ifb DIR_OR_JSON [...] --pd DIR_OR_JSON [...] [--ifb-nodes 1] [--pd-nodes 2] [--gpus-per-node 8]
                       [--cp 0.7603 0.7526 --cd 0.9413] [--out report.md] [--json out.json]
  analyze_benchmark.py check FILE.json [--gate A|B]        exit 0 if that point passes the gate (used by bench_ladder.sh)

Metrics per concurrency point (from per-request data in the JSON):
  req/s = request_throughput; TTFT pXX = percentiles of `ttfts`;
  user tok/s = per-request 1/TPOT, TPOT = sum(itls)/(output_len-1); "user p50" = median, "user p99" = 1st percentile
  (the slowest 1% of users; higher is better).  MTP acceptance from the JSON when the server reported it.
Gates (defaults = project SLOs):  A: TTFT p95 <= 10 s and user p99 > 30 tok/s
                                   B: TTFT p50 < 5 s and user p50 > 55 and user p99 > 25 tok/s
A point is usable only if completed == num_prompts >= --min-requests, failed == 0, and the file name does not contain
invalid / smoke / warmup.  The reported capacity of a deployment per gate = highest passing concurrency.
"""
import argparse, glob, json, math, sys
from pathlib import Path

GATES = {
    "A": dict(ttft_key="p95", ttft_max=10.0, ttft_strict=False, user_p50=None, user_p99=30.0),
    "B": dict(ttft_key="p50", ttft_max=5.0, ttft_strict=True, user_p50=55.0, user_p99=25.0),
}
BAD_NAMES = ("invalid", "smoke", "warmup")

def pct(xs, q):  # numpy-compatible linear interpolation percentile
    s = sorted(xs); n = len(s)
    if n == 1: return s[0]
    k = (n - 1) * q / 100.0; f = math.floor(k); c = min(f + 1, n - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

def load_point(path):
    d = json.loads(Path(path).read_text())
    n = len(d.get("ttfts", []))
    out_lens, itls = d.get("output_lens", []), d.get("itls", [])
    users = []
    for itl, ol in zip(itls, out_lens):
        if ol and ol > 1 and itl:
            tpot = sum(itl) / (ol - 1)
            if tpot > 0: users.append(1.0 / tpot)
    t = d.get("ttfts", [])
    return dict(file=str(path), c=int(d.get("max_concurrency") or 0), n=n, completed=d.get("completed", n), failed=d.get("failed", 0),
        num_prompts=d.get("num_prompts"), req_s=d.get("request_throughput"), out_tok_s=d.get("output_throughput"),
        ttft_p50=pct(t, 50), ttft_p95=pct(t, 95), ttft_p99=pct(t, 99),
        user_p50=pct(users, 50) if users else None, user_p99=pct(users, 1) if users else None,   # slowest 1% of users
        mtp=(d.get("spec_decode_acceptance_rate")), duration=d.get("duration"))

def passes(pt, g, gates=GATES):
    G = gates[g]; ttft = pt["ttft_" + G["ttft_key"]]
    ok = ttft < G["ttft_max"] if G["ttft_strict"] else ttft <= G["ttft_max"]
    if pt["user_p99"] is None or pt["user_p99"] <= G["user_p99"]: ok = False
    if G["user_p50"] is not None and (pt["user_p50"] is None or pt["user_p50"] <= G["user_p50"]): ok = False
    return ok

def collect(paths, min_requests):
    files, skipped = [], []
    for p in paths:
        p = Path(p)
        files += [p] if p.is_file() else sorted(p.rglob("*.json"))
    pts = {}
    for f in files:
        if any(b in f.name.lower() for b in BAD_NAMES): skipped.append((str(f), "name")); continue
        try: pt = load_point(f)
        except Exception as e: skipped.append((str(f), f"unreadable: {e}")); continue
        if not pt["c"] or pt["n"] < min_requests or pt["failed"] or (pt["num_prompts"] and pt["completed"] != pt["num_prompts"]):
            skipped.append((str(f), f"c={pt['c']} n={pt['n']} failed={pt['failed']} (needs >= {min_requests} requests, 0 failed, all completed)")); continue
        if pt["c"] in pts and pts[pt["c"]]["n"] >= pt["n"]: skipped.append((str(f), f"duplicate of C{pt['c']}")); continue
        pts[pt["c"]] = pt
    return dict(sorted(pts.items())), skipped

def capacity(pts, g):
    ok = [c for c, p in pts.items() if passes(p, g)]
    if not ok: return None, []
    best = max(ok); warn = [c for c in pts if c < best and not passes(pts[c], g)]
    return pts[best], warn

def fmt(x, nd=3): return "-" if x is None else f"{x:.{nd}f}"

def table(name, pts):
    L = [f"### {name}: all valid points", "", "| C | req/s | out tok/s | TTFT p50/p95/p99 (s) | user p50 / p99 tok/s | MTP acc. | Gate A | Gate B |", "|---:|---:|---:|---|---|---:|:---:|:---:|"]
    for c, p in pts.items():
        mtp = f"{p['mtp']:.1f}%" if p["mtp"] is not None else "-"
        L.append(f"| {c} | {fmt(p['req_s'])} | {fmt(p['out_tok_s'],1)} | {fmt(p['ttft_p50'],2)} / {fmt(p['ttft_p95'],2)} / {fmt(p['ttft_p99'],2)} | {fmt(p['user_p50'],1)} / {fmt(p['user_p99'],1)} | {mtp} | {'PASS' if passes(p,'A') else 'FAIL'} | {'PASS' if passes(p,'B') else 'FAIL'} |")
    return L

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("file"); ap.add_argument("--gate", choices=["A", "B"], required=True); a = ap.parse_args()
        pt = load_point(a.file); ok = passes(pt, a.gate)
        print(f"C{pt['c']} req/s={fmt(pt['req_s'])} ttft p50/p95={fmt(pt['ttft_p50'],2)}/{fmt(pt['ttft_p95'],2)} user p50/p99={fmt(pt['user_p50'],1)}/{fmt(pt['user_p99'],1)} gate {a.gate}: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ifb", nargs="+", required=True); ap.add_argument("--pd", nargs="+", required=True)
    ap.add_argument("--ifb-nodes", type=int, default=1); ap.add_argument("--pd-nodes", type=int, default=2); ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--min-requests", type=int, default=100); ap.add_argument("--cp", type=float, nargs="+", help="single-Prefill capacity req/s; one value, or two (Gate A, Gate B)"); ap.add_argument("--cd", type=float, nargs="+", help="single-Decode capacity req/s; one value, or two (Gate A, Gate B)")
    ap.add_argument("--out"); ap.add_argument("--json"); a = ap.parse_args()
    ifb, sk1 = collect(a.ifb, a.min_requests); pd, sk2 = collect(a.pd, a.min_requests)
    if not ifb or not pd: sys.exit(f"no valid points (IFB {len(ifb)}, PD {len(pd)}). Skipped: {sk1 + sk2}")
    gi, gp = a.ifb_nodes * a.gpus_per_node, a.pd_nodes * a.gpus_per_node
    res = {"gates": {}, "ifb_points": ifb, "pd_points": pd}
    L = ["# P/D vs IFB benchmark report", "",
         f"IFB: {a.ifb_nodes} node(s) / {gi} GPU. P/D: {a.pd_nodes} node(s) / {gp} GPU. Valid points: IFB {len(ifb)}, P/D {len(pd)} (N >= {a.min_requests}, 0 failed).", ""]
    L += ["## Gain at each SLO gate", "", "| Metric | Gate A: IFB | Gate A: P/D | Gain A | Gate B: IFB | Gate B: P/D | Gain B |", "|---|---:|---:|---:|---:|---:|---:|"]
    cap = {}
    for g in "AB":
        bi, wi = capacity(ifb, g); bp, wp = capacity(pd, g)
        if not bi or not bp: L += ["", f"WARNING: no point passes Gate {g} for {'IFB' if not bi else 'P/D'}; gains for that gate are not computed."]; continue
        cap[g] = (bi, bp, wi, wp)
        res["gates"][g] = dict(ifb_c=bi["c"], pd_c=bp["c"])
    def row(label, f, nd=3, gain="ratio"):
        cells = []
        for g in "AB":
            if g not in cap: cells += ["-", "-", "-"]; continue
            bi, bp, *_ = cap[g]; vi, vp = f(bi, "ifb"), f(bp, "pd")
            gv = (vp / vi - 1) * 100 if (vi and vp is not None and gain == "ratio") else (vp - vi if gain == "diff" else None)
            gs = "-" if gv is None else (f"{gv:+.1f}%" if gain == "ratio" else f"{gv:+.2f} s")
            cells += [fmt(vi, nd), fmt(vp, nd), gs]
        return f"| {label} | " + " | ".join(cells) + " |"
    rows = [("Max passing concurrency", lambda p, k: float(p["c"]), 0, "none"),
            ("Deployment req/s", lambda p, k: p["req_s"], 3, "ratio"),
            ("Deployment output tok/s", lambda p, k: p["out_tok_s"], 1, "ratio"),
            ("**Per-node req/s**", lambda p, k: p["req_s"] / (a.ifb_nodes if k == "ifb" else a.pd_nodes), 3, "ratio"),
            ("**Output tok/s/GPU**", lambda p, k: p["out_tok_s"] / (gi if k == "ifb" else gp), 2, "ratio"),
            ("User output p50 (tok/s)", lambda p, k: p["user_p50"], 1, "ratio"),
            ("User output p99 (tok/s)", lambda p, k: p["user_p99"], 1, "ratio"),
            ("TTFT p50 (s)", lambda p, k: p["ttft_p50"], 2, "diff"),
            ("TTFT p95 (s)", lambda p, k: p["ttft_p95"], 2, "diff")]
    for label, f, nd, gain in rows: L.append(row(label, f, nd, gain))
    for g in "AB":
        if g in cap:
            bi, bp, wi, wp = cap[g]
            for nm, w in (("IFB", wi), ("P/D", wp)):
                if w: L.append(f"\nNote: Gate {g} {nm} is non-monotonic: lower concurrency point(s) {w} fail while a higher one passes - re-check those points.")
            res["gates"][g].update(deploy_gain_pct=(bp["req_s"] / bi["req_s"] - 1) * 100,
                per_node_gain_pct=((bp["req_s"] / a.pd_nodes) / (bi["req_s"] / a.ifb_nodes) - 1) * 100,
                tok_s_per_gpu_gain_pct=((bp["out_tok_s"] / gp) / (bi["out_tok_s"] / gi) - 1) * 100)
    L += ["", "Deployment gain compares total throughput (P/D uses more nodes). **Per-node / per-GPU gain is the fair efficiency comparison**; report both."]
    if a.cp and a.cd:
        cps = {"A": a.cp[0], "B": a.cp[-1]}; cds = {"A": a.cd[0], "B": a.cd[-1]}
        L += ["", "## Achievement vs single-pool theory (1P1D)", ""]
        for g in "AB":
            if g in cap and a.pd_nodes == 2:
                up = min(cps[g], cds[g]); L.append(f"- Gate {g}: C_P={cps[g]}, C_D={cds[g]} -> 1P1D upper bound {up:.3f} req/s; measured {cap[g][1]['req_s']:.3f} = {cap[g][1]['req_s']/up*100:.0f}% (gap = KV transfer, routing, queueing, sync).")
    L += [""] + table("IFB", ifb) + [""] + table("P/D", pd)
    if sk1 or sk2: L += ["", "## Skipped files", ""] + [f"- {f}: {why}" for f, why in sk1 + sk2]
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out: Path(a.out).write_text(txt)
    if a.json: Path(a.json).write_text(json.dumps(res, indent=1, default=str))

if __name__ == "__main__": main()
