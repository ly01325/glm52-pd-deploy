#!/usr/bin/env python3
"""Verify a GLM-5.2 P/D deployment from worker logs (Handbook s7.2/7.3/8.3).

usage: verify_deployment.py --prefill-log P.log --decode-log D.log [--mtp-acceptance 95.6 --ifb-acceptance 94.54]
Exit 0 = all required checks pass; 1 = at least one FAIL.
"""
import argparse, re, sys

ERR = re.compile(r"Traceback \(most recent|CUDA error|out of memory|CUDA out of memory|EngineDeadError|NCCL WARN.*(timeout|error)", re.I)

def load(p):
    try: return open(p, errors="replace").read()
    except Exception as e: print(f"FAIL cannot read {p}: {e}"); sys.exit(1)

def chk(name, ok, detail="", req=True):
    tag = "PASS" if ok else ("FAIL" if req else "WARN")
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok or not req

def common(txt, role):
    r = []
    m = re.search(r"Initializing a V1 LLM engine \(v([\d.]+)\)", txt)
    r.append(chk("vLLM engine version found", bool(m), f"v{m.group(1)}" if m else "not found"))
    r.append(chk("MTP speculative decoding enabled", "SpeculativeConfig(method='mtp'" in txt))
    r.append(chk("Prefix caching enabled in EngineConfig", "enable_prefix_caching=True" in txt, "if you intended it OFF use --no-enable-prefix-caching", req=False))
    r.append(chk("NIXL KV connector present", bool(re.search(r"nixl", txt, re.I))))
    r.append(chk("no Traceback / CUDA error / OOM", not ERR.search(txt), (ERR.search(txt).group(0) if ERR.search(txt) else "")))
    ep = re.search(r"Using \[([^\]]*)\] all-reduce backends.*group 'ep:0'", txt)
    print(f"  [INFO] EP group backend: {ep.group(1) if ep else 'not logged'} (PYNCCL expected; NCCL baseline still matters)")
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefill-log"); ap.add_argument("--decode-log")
    ap.add_argument("--mtp-acceptance", type=float); ap.add_argument("--ifb-acceptance", type=float, default=94.54)
    a = ap.parse_args(); res = []
    if a.prefill_log:
        print(f"Prefill: {a.prefill_log}"); t = load(a.prefill_log); res += common(t, "prefill")
        res.append(chk("B12X PCIe DMA all-reduce enabled", "B12x PCIe DMA all-reduce enabled" in t))
        res.append(chk("TP group dispatch includes B12X_PCIE_DMA", bool(re.search(r"Using \['B12X_PCIE_DMA'[^\]]*\].*group 'tp:0'", t)), "", req=False))
        res.append(chk("FlashInfer IPC NOT used for TP on Prefill", not re.search(r"Using \['FLASHINFER_PCIE_IPC'[^\]]*\].*group 'tp:0'", t)))
    if a.decode_log:
        print(f"Decode: {a.decode_log}"); t = load(a.decode_log); res += common(t, "decode")
        res.append(chk("TP group uses FLASHINFER_PCIE_IPC", bool(re.search(r"Using \['FLASHINFER_PCIE_IPC'[^\]]*\].*group 'tp:0'", t))))
        m = re.search(r"cudagraph_mode'?:\s*<CUDAGraphMode\.(\w+)", t)
        res.append(chk("CUDA Graph mode is PIECEWISE", bool(m) and m.group(1) == "PIECEWISE",
                       f"found {m.group(1) if m else 'none'}; FULL / FULL_AND_PIECEWISE breaks MTP in P/D (acceptance ~14-15%)"))
    if a.mtp_acceptance is not None:
        res.append(chk(f"MTP acceptance {a.mtp_acceptance:.2f}% >= IFB {a.ifb_acceptance:.2f}% - 1pp", a.mtp_acceptance >= a.ifb_acceptance - 1.0))
    if not (a.prefill_log or a.decode_log): ap.error("give at least one log")
    ok = all(res); print("\nRESULT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
