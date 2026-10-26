#!/usr/bin/env python3
"""P/D ratio from single-pool capacities (Handbook s6).  C_P, C_D in req/s at the SLO boundary.

usage: compute_pd_ratio.py --cp 0.7603 --cd 0.9413 [--headroom 0.2] [--max-p 16]
"""
import argparse
from fractions import Fraction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=float, required=True, help="single Prefill capacity, req/s")
    ap.add_argument("--cd", type=float, required=True, help="single Decode capacity, req/s")
    ap.add_argument("--headroom", type=float, default=0.2, help="spare capacity on the bottleneck pool (0.15-0.30)")
    ap.add_argument("--max-p", type=int, default=16)
    a = ap.parse_args()
    r = a.cp / a.cd
    print(f"theory N_D/N_P = C_P/C_D = {r:.4f}   (1P : {r:.2f}D)")
    print(f"bottleneck: {'Prefill' if r < 1 else 'Decode'} (system capacity = min(N_P*C_P, N_D*C_D))")
    print(f"\n{'P':>3} {'D':>3} {'lambda_max':>11} {'util P':>7} {'util D':>7} {'req/s/node':>11}")
    seen = []
    for p in range(1, a.max_p + 1):
        for d in sorted({max(1, int(r * p)), max(1, int(r * p) + 1), max(1, round(r * p))}):
            if (p, d) in seen: continue
            seen.append((p, d)); lam = min(p * a.cp, d * a.cd)
            print(f"{p:>3} {d:>3} {lam:>11.3f} {lam/(p*a.cp):>7.0%} {lam/(d*a.cd):>7.0%} {lam/(p+d):>11.3f}")
    fr = Fraction(r).limit_denominator(6)
    print(f"\nsmall integer ratio near theory: {fr.denominator}P:{fr.numerator}D (D/P={float(fr):.3f})")
    print(f"plan with >= {a.headroom:.0%} headroom: size bottleneck pool for demand/(1-{a.headroom:.2f}); then VALIDATE neighbouring integer ratios end-to-end (KV transfer/router costs are not in this model).")

if __name__ == "__main__": main()
