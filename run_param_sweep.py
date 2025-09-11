#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
from mt_upe_core import simulate_hybrid_avg

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

EXPECTED_FIELDS = [
    "trials","mode","curvature","loss_mm","distance_um","win_ns",
    "emit_photons_mean","emit_photons_sem",
    "gate_photons_mean","gate_photons_sem",
    "coincidence_ratio_mean","coincidence_ratio_sem",
    "peak_sharpness_mean","peak_sharpness_sem",
    "p2rms_mean","p2rms_sem",
    "det_photons_mean","det_photons_sem",
    "snr_mean","snr_sem",
    "motor_net_steps_mean","motor_net_steps_sem",
    "photon_peak_mean","photon_peak_sem",
    "cilia_phase_reset_mean","cilia_phase_reset_sem",
    "spindle_metric_mean","spindle_metric_sem"
]

def header_mismatch(cols): return list(cols) != EXPECTED_FIELDS

def build_grid(args):
    if args.preset == "sweetspot":
        curvatures = [0.8,0.9,1.0,1.1]
        losses     = [3,5,7,10,15]
        distances  = [5,10,20,35,50]
    else:
        curvatures = [float(x) for x in args.curvatures.split(",") if x.strip()]
        losses     = [float(x) for x in args.losses.split(",") if x.strip()]
        distances  = [float(x) for x in args.distances.split(",") if x.strip()]
    return [(k,mu,r) for k in curvatures for mu in losses for r in distances]

def main():
    ap = argparse.ArgumentParser(description="Parameter-Sweep MT/UPE (HYP vs ANT, Gate-Optimierung, Live-Logs)")
    ap.add_argument("--preset", type=str, default="default", choices=["default","sweetspot"])
    ap.add_argument("--curvatures", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--losses", type=str, default="5,10,30,100")
    ap.add_argument("--distances", type=str, default="10,50,100")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--window-ns", type=float, default=0.5)
    ap.add_argument("--optimize-window", type=str, default="",
                    help="z.B. '0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5'")
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)
    ap.add_argument("--qe", type=float, default=0.6)
    ap.add_argument("--eta-geom", type=float, default=0.1)
    ap.add_argument("--dark-cps", type=float, default=100.0)
    ap.add_argument("--emission-scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="param_sweep_results.csv")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    win_list = [args.window_ns] if not args.optimize_window else \
               [float(x) for x in args.optimize_window.split(",") if x.strip()]
    ds = args.deadtime_ns*1e-9
    out_path = Path(args.out)

    mode = "a" if out_path.exists() else "w"
    if out_path.exists():
        try:
            cols = list(pd.read_csv(out_path, nrows=0).columns)
            if header_mismatch(cols):
                bak = out_path.with_suffix(out_path.suffix+".bak")
                out_path.rename(bak); print(f"[Info] Header-Mismatch. Backup: {bak.name}")
                mode = "w"
        except Exception:
            bak = out_path.with_suffix(out_path.suffix+".bak")
            out_path.rename(bak); print(f"[Info] CSV unlesbar. Backup: {bak.name}")
            mode = "w"
    write_header = (mode=="w") and (not args.no_header)

    grid = build_grid(args)
    total = len(grid)

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        if write_header: writer.writeheader()

        with tqdm(total=total, desc="Sweep", unit="combo") as pbar:
            for (curv, mu, r) in grid:
                def inner_tqdm(it, **kw): return tqdm(it, **kw)

                # Hypothese: bestes Gate suchen (max SNR)
                best = None
                best_win = None
                for wn in (win_list or [args.window_ns]):
                    res = simulate_hybrid_avg(
                        curv, "hypothesis", mu, r, trials=args.trials, base_seed=args.seed,
                        window_s=wn*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                        qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                        emission_scale=args.emission_scale, pbar=inner_tqdm
                    )
                    if (best is None) or (res["snr_mean"] > best["snr_mean"]):
                        best = res; best_win = wn

                # Live-Log HYP
                tqdm.write(
                    f"[HYP] κ={curv}, μ={mu} mm^-1, r={r} µm, win={best_win} ns | "
                    f"SNR={best['snr_mean']:.2f}, CR={best.get('coincidence_ratio_mean',0):.3f}, "
                    f"p2rms={best.get('p2rms_mean',0):.2f}, gateN={best.get('gate_photons_mean',0):.3f}"
                )

                row_h = {k: best.get(k, "") for k in EXPECTED_FIELDS}
                row_h["win_ns"] = float(best_win)
                writer.writerow(row_h); f.flush()

                # Antithese: identisches Gate
                res_a = simulate_hybrid_avg(
                    curv, "antithesis", mu, r, trials=args.trials, base_seed=args.seed,
                    window_s=best_win*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                    qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                    emission_scale=args.emission_scale, pbar=inner_tqdm
                )

                # Live-Log ANT
                tqdm.write(
                    f"[ANT] κ={curv}, μ={mu} mm^-1, r={r} µm, win={best_win} ns | "
                    f"SNR={res_a['snr_mean']:.2f}, CR={res_a.get('coincidence_ratio_mean',0):.3f}, "
                    f"p2rms={res_a.get('p2rms_mean',0):.2f}, gateN={res_a.get('gate_photons_mean',0):.3f}"
                )

                row_a = {k: res_a.get(k, "") for k in EXPECTED_FIELDS}
                row_a["win_ns"] = float(best_win)
                writer.writerow(row_a); f.flush()

                pbar.set_postfix({
                    "κ":curv,"μ(mm^-1)":mu,"r(µm)":r,"win(ns)":best_win,
                    "SNR(HYP)":f"{best['snr_mean']:.2f}","SNR(ANT)":f"{res_a['snr_mean']:.2f}"
                })
                pbar.update(1)

    # Kurz-Report (gleiches Gate)
    try:
        df = pd.read_csv(out_path)
        print("\n=== DISKRIMINATIONS-FÄLLE (HYP >= 5σ & ANT < 5σ; gleiches Gate) ===")
        required = ["mode","curvature","loss_mm","distance_um","win_ns","snr_mean"]
        for c in required:
            if c not in df.columns:
                raise RuntimeError(f"Spalte fehlt: '{c}'. Spalten: {list(df.columns)}")
        keys = ["curvature","loss_mm","distance_um","win_ns"]
        df["_ord"] = np.arange(len(df))
        sel = (df.sort_values(["snr_mean","_ord"])
                 .groupby(keys+["mode"], as_index=False)
                 .tail(1))
        hyp = sel[sel["mode"]=="hypothesis"].set_index(keys)
        ant = sel[sel["mode"]=="antithesis"].set_index(keys)
        common = hyp.index.intersection(ant.index)

        any_case=False
        for idx in common:
            h, a = hyp.loc[idx], ant.loc[idx]
            if float(h["snr_mean"])>=5.0 and float(a["snr_mean"])<5.0:
                kappa, mu, r, win = idx
                print(f"κ={kappa}, μ={mu} mm^-1, r={r} µm, Gate={win} ns"
                      f"  -->  HYP SNR={float(h['snr_mean']):.2f}, ANT SNR={float(a['snr_mean']):.2f}")
                any_case=True
        if not any_case: print("(keine Fälle mit HYP ≥5σ und ANT <5σ gefunden)")
        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")

    except Exception as e:
        print(f"\nHinweis: Auswertung nicht möglich: {e}")
        try:
            print("Spalten:", list(pd.read_csv(out_path, nrows=0).columns))
        except Exception:
            pass
        print(f"Rohdaten in: {out_path.resolve()}")

if __name__ == "__main__":
    main()
