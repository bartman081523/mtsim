#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from mt_upe_core import simulate_hybrid_avg

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

def main():
    ap = argparse.ArgumentParser(description="Hybrid-Einzelsim (HYP vs ANT) mit Peak-Gating")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--loss-mm", type=float, default=10.0)
    ap.add_argument("--r-um", type=float, default=50.0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--win-ns", type=float, default=0.5, help="Gate-Breite in ns (sub-ns möglich)")
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)
    ap.add_argument("--qe", type=float, default=0.7)
    ap.add_argument("--eta-geom", type=float, default=0.2)
    ap.add_argument("--dark-cps", type=float, default=50.0)
    ap.add_argument("--emission-scale", type=float, default=1.0)
    args = ap.parse_args()

    win_s = args.win_ns * 1e-9
    ds = args.deadtime_ns * 1e-9

    for mode in ["hypothesis","antithesis"]:
        res = simulate_hybrid_avg(
            curvature=args.curvature, mode=mode, loss_mm=args.loss_mm, r_um=args.r_um,
            trials=args.trials, base_seed=args.seed, window_s=win_s, deadtime_s=ds,
            afterpulse=args.afterpulse, qe=args.qe, eta_geom=args.eta_geom,
            dark_cps=args.dark_cps, emission_scale=args.emission_scale,
            pbar=tqdm
        )
        print(res)

if __name__ == "__main__":
    main()
