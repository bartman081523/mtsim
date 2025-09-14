#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from mt_upe_core import simulate_hybrid_avg
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

def fmt(x, nd=2, fallback="-"):
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return fallback

def run_mode(args, mode: str):
    res = simulate_hybrid_avg(
        curvature=args.curvature, mode=mode, loss_mm=args.loss_mm, r_um=args.r_um,
        trials=args.trials, base_seed=args.seed,
        window_s=args.win_ns*1e-9, deadtime_s=args.deadtime_ns*1e-9, afterpulse=args.afterpulse,
        qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps, emission_scale=args.emission_scale,
        poissonize=args.poissonize,
        # Wichtig: keine 'desc' hier setzen – das macht die Core-Funktion bereits.
        pbar=lambda it, **kw: tqdm(it, **kw)
    )
    return res

def main():
    ap = argparse.ArgumentParser(description="Stufe 3 – Hybrid-Dynamik (HYP vs ANT: SNR, Score, ΔR, etc.)")
    ap.add_argument("--mode", type=str, default="compare", choices=["hypothesis","antithesis","compare"])
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--loss-mm", type=float, default=7.0)
    ap.add_argument("--r-um", type=float, default=20.0)

    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--win-ns", type=float, default=0.2)
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)

    ap.add_argument("--qe", type=float, default=0.8)
    ap.add_argument("--eta-geom", type=float, default=0.30)
    ap.add_argument("--dark-cps", type=float, default=20.0)
    ap.add_argument("--emission-scale", type=float, default=3.0)
    ap.add_argument("--poissonize", action="store_true", help="Poisson-Zählsampling (inkl. Dark) aktivieren")

    args = ap.parse_args()

    print("\n=== HYBRID DYNAMICS COMPARISON ===")

    if args.mode in ("hypothesis", "compare"):
        res_h = run_mode(args, "hypothesis")
        print(
            "HYP: "
            f"SNR={fmt(res_h.get('snr_mean'))}, "
            f"CR={fmt(res_h.get('coincidence_ratio_mean'),3)}, "
            f"p2r={fmt(res_h.get('p2rms_mean'))}, "
            f"score={fmt(res_h.get('score_mean'))}, "
            f"R_before={fmt(res_h.get('R_before_mean'))}, "
            f"R_after={fmt(res_h.get('R_after_mean'))}, "
            f"ΔR={fmt(res_h.get('sync_gain_mean'),3)}"
        )
    if args.mode in ("antithesis", "compare"):
        res_a = run_mode(args, "antithesis")
        print(
            "ANT: "
            f"SNR={fmt(res_a.get('snr_mean'))}, "
            f"CR={fmt(res_a.get('coincidence_ratio_mean'),3)}, "
            f"p2r={fmt(res_a.get('p2rms_mean'))}, "
            f"score={fmt(res_a.get('score_mean'))}, "
            f"R_before={fmt(res_a.get('R_before_mean'))}, "
            f"R_after={fmt(res_a.get('R_after_mean'))}, "
            f"ΔR={fmt(res_a.get('sync_gain_mean'),3)}"
        )

    if args.mode == "compare":
        dsnr = None
        dscore = None
        dR = None
        try:
            dsnr = (res_h["snr_mean"] - res_a["snr_mean"])
        except Exception:
            pass
        try:
            dscore = (res_h["score_mean"] - res_a["score_mean"])
        except Exception:
            pass
        try:
            dR = (res_h["sync_gain_mean"] - res_a["sync_gain_mean"])
        except Exception:
            pass

        if dsnr is not None:
            print(f"\nΔSNR = {fmt(dsnr)}")
        if dscore is not None:
            print(f"ΔScore = {fmt(dscore)}")
        if dR is not None:
            print(f"Δ(ΔR) = {fmt(dR,3)}")

if __name__ == "__main__":
    main()
