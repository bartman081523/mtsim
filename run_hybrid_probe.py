#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, List, Tuple
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

# Import: nimmt an, dass mt_upe_core im selben Projekt liegt
from mt_upe_core import simulate_hybrid_once

# ----------------------------- Helpers ---------------------------------

def mean_sem(arr: List[float]) -> Tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a) > 1 else 0.0
    return m, s

def run_trials(mode: str,
               curvature: float, loss_mm: float, r_um: float,
               trials: int, seed0: int,
               window_ns: float, deadtime_ns: float, afterpulse: float,
               qe: float, eta_geom: float, dark_cps: float, emission_scale: float,
               poissonize: bool) -> Dict[str, float]:
    """
    Führt 'trials' Einzelläufe durch, loggt Live-Postfix und liefert Summary (Mean±SEM).
    """
    # Akkus
    snr_list, cr_list, p2r_list, gateN_list = [], [], [], []
    peak_list, steps_list = [], [],

    # Zeitskalen
    window_s = float(window_ns) * 1e-9
    deadtime_s = float(deadtime_ns) * 1e-9

    # Trial-Loop
    with tqdm(range(trials),
              desc=f"{mode.upper()}",
              total=trials,
              unit="trial") as pbar:

        for i in pbar:
            seed = seed0 + i

            # simulate_hybrid_once – robust gegen ältere Signaturen (poissonize optional)
            try:
                row = simulate_hybrid_once(
                    curvature=curvature, mode=mode, loss_mm=loss_mm, r_um=r_um, seed=seed,
                    window_s=window_s, deadtime_s=deadtime_s, afterpulse=afterpulse,
                    qe=qe, eta_geom=eta_geom, dark_cps=dark_cps, emission_scale=emission_scale,
                    poissonize=poissonize
                )
            except TypeError:
                # Fallback, falls das Core-Modul (noch) kein 'poissonize' kennt
                row = simulate_hybrid_once(
                    curvature=curvature, mode=mode, loss_mm=loss_mm, r_um=r_um, seed=seed,
                    window_s=window_s, deadtime_s=deadtime_s, afterpulse=afterpulse,
                    qe=qe, eta_geom=eta_geom, dark_cps=dark_cps, emission_scale=emission_scale
                )

            # Extrahiere Metriken
            snr = float(row.get("snr", 0.0))
            cr  = float(row.get("coincidence_ratio", 0.0))
            p2r = float(row.get("p2rms", 0.0))
            gN  = float(row.get("gate_photons", 0.0))
            peak = float(row.get("photon_peak", 0.0))
            steps = float(row.get("motor_net_steps", 0.0))

            snr_list.append(snr)
            cr_list.append(cr)
            p2r_list.append(p2r)
            gateN_list.append(gN)
            peak_list.append(peak)
            steps_list.append(steps)

            # Live-Score (robust gegen Poisson/Deadtime-Effekte)
            score = cr * p2r
            pbar.set_postfix({
                "snr": f"{snr:.2f}",
                "CR": f"{cr:.3f}",
                "p2r": f"{p2r:.2f}",
                "gateN": f"{gN:.1f}",
                "score": f"{score:.2f}"
            })

    # Summary
    snr_m,  snr_s  = mean_sem(snr_list)
    cr_m,   cr_s   = mean_sem(cr_list)
    p2r_m,  p2r_s  = mean_sem(p2r_list)
    gN_m,   gN_s   = mean_sem(gateN_list)
    peak_m, peak_s = mean_sem(peak_list)
    step_m, step_s = mean_sem(steps_list)

    score_m, score_s = mean_sem(np.array(cr_list)*np.array(p2r_list))

    print(
        f"\n[{mode.upper()} SUMMARY] Gate={window_ns:.3f} ns  |  "
        f"SNR={snr_m:.2f}±{snr_s:.2f},  "
        f"CR={cr_m:.3f}±{cr_s:.3f},  "
        f"p2r={p2r_m:.2f}±{p2r_s:.2f},  "
        f"gateN={gN_m:.1f}±{gN_s:.1f},  "
        f"score={score_m:.2f}±{score_s:.2f}"
    )

    return {
        "snr_m": snr_m, "snr_s": snr_s,
        "cr_m": cr_m, "cr_s": cr_s,
        "p2r_m": p2r_m, "p2r_s": p2r_s,
        "gateN_m": gN_m, "gateN_s": gN_s,
        "peak_m": peak_m, "peak_s": peak_s,
        "steps_m": step_m, "steps_s": step_s,
        "score_m": score_m, "score_s": score_s
    }

# ------------------------------ Main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Hybrid-Probe (HYP/ANT/COMPARE) mit Score=CR*p2rms")
    ap.add_argument("--mode", type=str, default="compare",
                    choices=["hyp", "ant", "compare"],
                    help="Einzelmodus oder Vergleich beider Modi")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--loss-mm", type=float, default=7.0)
    ap.add_argument("--r-um", type=float, default=20.0)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)

    # Detektor/Gate
    ap.add_argument("--win-ns", type=float, default=0.2)
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)

    # Hardware
    ap.add_argument("--qe", type=float, default=0.8)
    ap.add_argument("--eta-geom", type=float, default=0.30)
    ap.add_argument("--dark-cps", type=float, default=20.0)

    # Emissions-Skalierung
    ap.add_argument("--emission-scale", type=float, default=3.0)

    # Realismus-Schalter
    ap.add_argument("--poissonize", action="store_true",
                    help="Poisson-Shotnoise & Darkcount-Sampling pro Trial aktivieren")

    args = ap.parse_args()

    if args.mode in ("hyp", "compare"):
        resH = run_trials(
            mode="hypothesis",
            curvature=args.curvature, loss_mm=args.loss_mm, r_um=args.r_um,
            trials=args.trials, seed0=args.seed,
            window_ns=args.win_ns, deadtime_ns=args.deadtime_ns, afterpulse=args.afterpulse,
            qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
            emission_scale=args.emission_scale, poissonize=args.poissonize
        )
    if args.mode in ("ant", "compare"):
        resA = run_trials(
            mode="antithesis",
            curvature=args.curvature, loss_mm=args.loss_mm, r_um=args.r_um,
            trials=args.trials, seed0=args.seed,
            window_ns=args.win_ns, deadtime_ns=args.deadtime_ns, afterpulse=args.afterpulse,
            qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
            emission_scale=args.emission_scale, poissonize=args.poissonize
        )

    if args.mode == "compare":
        dsnr = resH["snr_m"] - resA["snr_m"]
        ratio = (resH["snr_m"] / resA["snr_m"]) if resA["snr_m"] > 0 else 0.0

        dscore = resH["score_m"] - resA["score_m"]
        ratio_score = (resH["score_m"] / resA["score_m"]) if resA["score_m"] > 0 else 0.0

        print(f"\n==> ΔSNR = {dsnr:.2f}   |   Ratio = {ratio:.2f}")
        print(f"==> ΔScore = {dscore:.2f}   |   Score-Ratio = {ratio_score:.2f}")

if __name__ == "__main__":
    main()
