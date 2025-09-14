#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, math
from pathlib import Path
from statistics import mean, pstdev
from mt_upe_core import simulate_hybrid_avg

SEEDS_DEFAULT = [1001, 1002, 2001, 2002, 3001]

def ci95(m, sem): return (m-1.96*sem, m+1.96*sem)

def run_config(args, mode: str, seed: int):
    return simulate_hybrid_avg(
        curvature=args.curvature, mode=mode, loss_mm=args.loss_mm, r_um=args.r_um,
        trials=args.trials, base_seed=seed,
        window_s=args.win_ns*1e-9, deadtime_s=args.deadtime_ns*1e-9, afterpulse=args.afterpulse,
        qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
        emission_scale=args.emission_scale, poissonize=args.poissonize,
        pbar=lambda it, **kw: it
    )

def main():
    ap = argparse.ArgumentParser(description="Stage 3 Evidence Report (fixed params, multi-seed)")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--loss-mm", type=float, default=7.0)
    ap.add_argument("--r-um", type=float, default=20.0)
    ap.add_argument("--win-ns", type=float, default=0.2)
    ap.add_argument("--qe", type=float, default=0.8)
    ap.add_argument("--eta-geom", type=float, default=0.30)
    ap.add_argument("--dark-cps", type=float, default=0.0)
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)
    ap.add_argument("--emission-scale", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS_DEFAULT)))
    ap.add_argument("--poissonize", action="store_true")
    ap.add_argument("--out", type=str, default="stage3_report.csv")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []

    print("\n=== STAGE 3 REPORT (fixed config, multi-seed) ===")
    print(f"κ={args.curvature}, μ={args.loss_mm} mm^-1, r={args.r_um} µm, win={args.win_ns} ns, "
          f"QE={args.qe}, η={args.eta_geom}, dark={args.dark_cps} cps, dead={args.deadtime_ns} ns, "
          f"ES={args.emission_scale}, trials/seed={args.trials}, seeds={seeds}, poisson={args.poissonize}\n")

    agg = {"H": [], "A": []}
    for mode in ("hypothesis","antithesis"):
        for sd in seeds:
            res = run_config(args, mode, sd)
            rec = {
                "mode": mode, "seed": sd,
                "snr": res.get("snr_mean", float("nan")),
                "score": res.get("score_mean", float("nan")),
                "R_before": res.get("R_before_mean", float("nan")),
                "R_after": res.get("R_after_mean", float("nan")),
                "dR": res.get("sync_gain_mean", float("nan")),
                "CR": res.get("coincidence_ratio_mean", float("nan")),
                "p2r": res.get("p2rms_mean", float("nan")),
                "gateN": res.get("gate_photons_mean", float("nan")),
            }
            rows.append(rec)
            agg["H" if mode=="hypothesis" else "A"].append(rec)

    # Aggregate über Seeds
    def msem(vals):
        m = mean(vals)
        sd = pstdev(vals) if len(vals)>1 else 0.0
        sem = sd / math.sqrt(len(vals)) if len(vals)>1 else 0.0
        return m, sem

    def collect(key):
        H = [r[key] for r in agg["H"]]
        A = [r[key] for r in agg["A"]]
        mH, seH = msem(H); mA, seA = msem(A)
        return (mH, seH, mA, seA, mH-mA)

    mSNR = collect("snr")
    mScore = collect("score")
    mRbef = collect("R_before")
    mRaft = collect("R_after")
    mdR = collect("dR")

    def line(lbl, pack, nd=3):
        mH,seH,mA,seA,delta = pack
        ciH = ci95(mH,seH); ciA = ci95(mA,seA)
        return (f"{lbl:8s}  "
                f"H={mH:.{nd}f}±{seH:.{nd}f} (95% {ciH[0]:.{nd}f}..{ciH[1]:.{nd}f})   "
                f"A={mA:.{nd}f}±{seA:.{nd}f} (95% {ciA[0]:.{nd}f}..{ciA[1]:.{nd}f})   "
                f"Δ={delta:.{nd}f}")

    print(line("SNR", mSNR, nd=2))
    print(line("Score", mScore))
    print(line("R_before", mRbef))
    print(line("R_after",  mRaft))
    print(line("ΔR", mdR, nd=3))

    # CSV schreiben
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[OK] Detailzeilen -> {out.resolve()}")
    print("[Done]")
if __name__ == "__main__":
    main()
