#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from mt_upe_core import simulate_hybrid_avg

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

# ---- Helper -----------------------------------------------------------------

def parse_list_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

EXPECTED_FIELDS = [
    "trials","mode","curvature","loss_mm","distance_um",
    "qe","eta_geom","dark_cps","deadtime_ns","win_ns",
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
        curvatures = parse_list_floats(args.curvatures)
        losses     = parse_list_floats(args.losses)
        distances  = parse_list_floats(args.distances)

    qes   = parse_list_floats(args.qe)
    etas  = parse_list_floats(args.eta_geom)
    darks = parse_list_floats(args.dark_cps)
    dts   = parse_list_floats(args.deadtime_ns)
    es_list = parse_list_floats(args.emission_scale)

    win_list = [args.window_ns] if not args.optimize_window else \
               parse_list_floats(args.optimize_window)

    grid = []
    for k in curvatures:
        for mu in losses:
            for r in distances:
                for qe in qes:
                    for eta in etas:
                        for dark in darks:
                            for dt_ns in dts:
                                for es in es_list:
                                    grid.append((k, mu, r, qe, eta, dark, dt_ns, tuple(win_list), es))
    return grid

# ---- Runner -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parameter-Sweep MT/UPE (HYP vs ANT, tqdm, CSV)")
    ap.add_argument("--preset", type=str, default="default", choices=["default","sweetspot"])
    ap.add_argument("--curvatures", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--losses", type=str, default="5,10,30,100")
    ap.add_argument("--distances", type=str, default="10,50,100")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)

    ap.add_argument("--window-ns", type=float, default=100.0)
    ap.add_argument("--optimize-window", type=str, default="", help="z.B. '0.1,0.2,0.5' in ns")
    ap.add_argument("--deadtime-ns", type=str, default="0.0", help="Kommagetrennte Liste (ns)")

    ap.add_argument("--afterpulse", type=float, default=0.0)

    ap.add_argument("--qe", type=str, default="0.6", help="Kommagetrennt (z.B. '0.7,0.8')")
    ap.add_argument("--eta-geom", type=str, default="0.1", help="Kommagetrennt (z.B. '0.2,0.3')")
    ap.add_argument("--dark-cps", type=str, default="100.0", help="Kommagetrennt (z.B. '20,50,100')")
    ap.add_argument("--emission-scale", type=str, default="1.0", help="Kommaliste (z.B. '1,2,3,5')")
    ap.add_argument("--poissonize", action="store_true", help="Poisson-Shotnoise + Darkcount-Sampling pro Trial")

    ap.add_argument("--out", type=str, default="param_sweep_results.csv")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

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
            for (curv, mu, r, qe, eta, dark, dt_ns, win_tuple, es) in grid:

                def inner_tqdm(it, **kw): return tqdm(it, **kw)

                # 1) Hypothese: bestes Gate (nach SNR) suchen
                best = None
                for wn in win_tuple:
                    res = simulate_hybrid_avg(
                        curv, "hypothesis", mu, r,
                        trials=args.trials, base_seed=args.seed,
                        window_s=wn*1e-9, deadtime_s=dt_ns*1e-9, afterpulse=args.afterpulse,
                        qe=qe, eta_geom=eta, dark_cps=dark, emission_scale=es,
                        poissonize=args.poissonize, pbar=inner_tqdm
                    )
                    if (best is None) or (res["snr_mean"] > best["snr_mean"]):
                        best = res; best["win_ns"] = wn

                print(f"[HYP] κ={curv}, μ={mu} mm^-1, r={r} µm, QE={qe}, η={eta}, dark={dark} cps, "
                      f"dead={dt_ns} ns, win={best['win_ns']} ns, es={es} | "
                      f"SNR={best['snr_mean']:.2f}, CR={best.get('coincidence_ratio_mean',0.0):.3f}, "
                      f"p2rms={best.get('p2rms_mean',0.0):.2f}, gateN={best.get('gate_photons_mean',0.0):.3f}")

                row_h = {k: "" for k in EXPECTED_FIELDS}
                row_h.update({
                    "trials": args.trials, "mode": "hypothesis",
                    "curvature": curv, "loss_mm": mu, "distance_um": r,
                    "qe": qe, "eta_geom": eta, "dark_cps": dark,
                    "deadtime_ns": dt_ns, "win_ns": best["win_ns"]
                })
                for k,v in best.items():
                    if k+"_mean" in EXPECTED_FIELDS: row_h[k+"_mean"] = v
                    if k+"_sem"  in EXPECTED_FIELDS: row_h[k+"_sem"]  = v
                writer.writerow(row_h); f.flush()

                # 2) Antithese mit identischem Gate
                res_a = simulate_hybrid_avg(
                    curv, "antithesis", mu, r,
                    trials=args.trials, base_seed=args.seed,
                    window_s=best["win_ns"]*1e-9, deadtime_s=dt_ns*1e-9, afterpulse=args.afterpulse,
                    qe=qe, eta_geom=eta, dark_cps=dark, emission_scale=es,
                    poissonize=args.poissonize, pbar=inner_tqdm
                )

                print(f"[ANT] κ={curv}, μ={mu} mm^-1, r={r} µm, QE={qe}, η={eta}, dark={dark} cps, "
                      f"dead={dt_ns} ns, win={best['win_ns']} ns, es={es} | "
                      f"SNR={res_a['snr_mean']:.2f}, CR={res_a.get('coincidence_ratio_mean',0.0):.3f}, "
                      f"p2rms={res_a.get('p2rms_mean',0.0):.2f}, gateN={res_a.get('gate_photons_mean',0.0):.3f}")

                row_a = {k: "" for k in EXPECTED_FIELDS}
                row_a.update({
                    "trials": args.trials, "mode": "antithesis",
                    "curvature": curv, "loss_mm": mu, "distance_um": r,
                    "qe": qe, "eta_geom": eta, "dark_cps": dark,
                    "deadtime_ns": dt_ns, "win_ns": best["win_ns"]
                })
                for k,v in res_a.items():
                    if k+"_mean" in EXPECTED_FIELDS: row_a[k+"_mean"] = v
                    if k+"_sem"  in EXPECTED_FIELDS: row_a[k+"_sem"]  = v
                writer.writerow(row_a); f.flush()

                pbar.set_postfix({
                    "κ": curv, "μ(mm^-1)": mu, "r(µm)": r,
                    "win(ns)": best["win_ns"], "es": es,
                    "SNR(HYP)": f"{best.get('snr_mean',0.0):.2f}",
                    "SNR(ANT)": f"{res_a.get('snr_mean',0.0):.2f}"
                })
                pbar.update(1)

    # Kurz-Report (gleiche Gates / gleiche HW)
    try:
        df = pd.read_csv(out_path)
        print("\n=== DISKRIMINATIONS-FÄLLE (HYP >= 5σ & ANT < 5σ; gleiches Gate & HW) ===")
        keys = ["curvature","loss_mm","distance_um","qe","eta_geom","dark_cps","deadtime_ns","win_ns"]
        hyp = df[df["mode"]=="hypothesis"].set_index(keys)
        ant = df[df["mode"]=="antithesis"].set_index(keys)
        common = hyp.index.intersection(ant.index)
        any_case=False
        for idx in common:
            h, a = hyp.loc[idx], ant.loc[idx]
            if h["snr_mean"]>=5.0 and a["snr_mean"]<5.0:
                print(f"κ={idx[0]}, μ={idx[1]} mm^-1, r={idx[2]} µm, QE={idx[3]}, η={idx[4]}, "
                      f"dark={idx[5]} cps, dead={idx[6]} ns, Gate={idx[7]} ns  "
                      f"-->  HYP SNR={h['snr_mean']:.2f}, ANT SNR={a['snr_mean']:.2f}")
                any_case=True
        if not any_case: print("(keine Fälle mit HYP ≥5σ und ANT <5σ gefunden)")
        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")
    except Exception as e:
        print(f"\nHinweis: Auswertung nicht möglich: {e}")
        print(f"Rohdaten in: {out_path.resolve()}")

if __name__ == "__main__":
    main()
