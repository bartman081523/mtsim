#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
from mt_upe_core import simulate_network_avg

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

EXPECTED_FIELDS = [
    "trials","mode","N","curvature","loss_mm","alpha","ell_um",
    "det_x_um","det_y_um","det_z_um","node_radius_um","kappa_sigma","win_ns",
    "emit_photons_total_mean","emit_photons_total_sem",
    "det_photons_total_mean","det_photons_total_sem",
    "gate_photons_mean","gate_photons_sem",
    "coincidence_ratio_mean","coincidence_ratio_sem",
    "peak_sharpness_mean","peak_sharpness_sem",
    "p2rms_mean","p2rms_sem",
    "snr_mean","snr_sem",
    "R_before_mean","R_before_sem",
    "R_after_mean","R_after_sem",
    "sync_gain_mean","sync_gain_sem",
    "mean_phase_reset_mean","mean_phase_reset_sem"
]

def header_mismatch(cols): return list(cols) != EXPECTED_FIELDS

def build_grid(args):
    if args.preset == "sweetspot":
        curvatures=[0.8,0.9,1.0,1.1]
        losses=[3,5,7,10,15]
        alphas=[0.1,0.2,0.3,0.5,0.8,1.0]
        ells=[5,8,12,20]
    else:
        curvatures=[float(x) for x in args.curvatures.split(",") if x.strip()]
        losses=[float(x) for x in args.losses.split(",") if x.strip()]
        alphas=[float(x) for x in args.alphas.split(",") if x.strip()]
        ells=[float(x) for x in args.ells_um.split(",") if x.strip()]
    return [(k,mu,a,ell) for k in curvatures for mu in losses for a in alphas for ell in ells]

def main():
    ap = argparse.ArgumentParser(description="Network MT/UPE Sweep (HYP vs ANT, globales Gate, Live-Logs)")
    ap.add_argument("--preset", type=str, default="default", choices=["default","sweetspot"])
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--node-radius-um", type=float, default=12.0)
    ap.add_argument("--det-pos-um", type=str, default="0,0,20")
    ap.add_argument("--curvatures", type=str, default="0.5,1.0")
    ap.add_argument("--losses", type=str, default="5,10,30")
    ap.add_argument("--alphas", type=str, default="0.1,0.3")
    ap.add_argument("--ells-um", type=str, default="8,12")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1000)

    ap.add_argument("--window-ns", type=float, default=0.5)
    ap.add_argument("--optimize-window", type=str, default="",
                    help="ps–subns: '0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5'")
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)
    ap.add_argument("--qe", type=float, default=0.6)
    ap.add_argument("--eta-geom", type=float, default=0.1)
    ap.add_argument("--dark-cps", type=float, default=100.0)
    ap.add_argument("--kappa-sigma", type=float, default=0.0)
    ap.add_argument("--emission-scale", type=float, default=1.0)
    ap.add_argument("--prc-eps", type=float, default=0.2,
                    help="PRC-Kopplungsstärke")

    ap.add_argument("--out", type=str, default="sweep_network.csv")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    det_pos = tuple(float(x) for x in args.det_pos_um.split(","))
    ds = args.deadtime_ns*1e-9
    win_list = [args.window_ns] if not args.optimize_window else \
               [float(x) for x in args.optimize_window.split(",") if x.strip()]

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

        with tqdm(total=total, desc="Network Sweep", unit="combo") as pbar:
            for (kappa, mu, alpha, ell) in grid:
                def inner_tqdm(it, **kw): return tqdm(it, **kw)

                # Hypothese: bestes Gate per SNR
                best = None; best_win = None
                for wn in win_list:
                    res_try = simulate_network_avg(
                        N=args.N, curvature=kappa, mode="hypothesis", loss_mm=mu,
                        det_pos_um=det_pos, node_radius_um=args.node_radius_um,
                        alpha=alpha, ell_um=ell, trials=args.trials, base_seed=args.seed,
                        window_s=wn*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                        qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                        kappa_sigma=args.kappa_sigma, emission_scale=args.emission_scale,
                        pbar=inner_tqdm, prc_eps=args.prc_eps
                    )
                    if (best is None) or (res_try["snr_mean"] > best["snr_mean"]):
                        best = res_try; best_win = wn

                # Live-Log HYP
                tqdm.write(
                    f"[HYP] N={args.N}, κ={kappa}, μ={mu}, α={alpha}, ℓ={ell} µm, win={best_win} ns | "
                    f"SNR={best['snr_mean']:.2f}, ΔR={best.get('sync_gain_mean',0):.3f}, "
                    f"CR={best.get('coincidence_ratio_mean',0):.3f}, p2rms={best.get('p2rms_mean',0):.2f}, "
                    f"gateN={best.get('gate_photons_mean',0):.3f}"
                )

                row_h = {k: best.get(k,"") for k in EXPECTED_FIELDS}
                row_h["win_ns"] = float(best_win)
                writer.writerow(row_h); f.flush()

                # Antithese: identisches Gate
                res_ant = simulate_network_avg(
                    N=args.N, curvature=kappa, mode="antithesis", loss_mm=mu,
                    det_pos_um=det_pos, node_radius_um=args.node_radius_um,
                    alpha=alpha, ell_um=ell, trials=args.trials, base_seed=args.seed,
                    window_s=best_win*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                    qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                    kappa_sigma=args.kappa_sigma, emission_scale=args.emission_scale,
                    pbar=inner_tqdm, prc_eps=args.prc_eps
                )

                # Live-Log ANT
                tqdm.write(
                    f"[ANT] N={args.N}, κ={kappa}, μ={mu}, α={alpha}, ℓ={ell} µm, win={best_win} ns | "
                    f"SNR={res_ant['snr_mean']:.2f}, ΔR={res_ant.get('sync_gain_mean',0):.3f}, "
                    f"CR={res_ant.get('coincidence_ratio_mean',0):.3f}, p2rms={res_ant.get('p2rms_mean',0):.2f}, "
                    f"gateN={res_ant.get('gate_photons_mean',0):.3f}"
                )

                row_a = {k: res_ant.get(k,"") for k in EXPECTED_FIELDS}
                row_a["win_ns"] = float(best_win)
                writer.writerow(row_a); f.flush()

                pbar.set_postfix({
                    "κ":kappa,"μ":mu,"α":alpha,"ℓ(µm)":ell,"win(ns)":best_win,
                    "SNR(H)":f"{best['snr_mean']:.2f}","SNR(A)":f"{res_ant['snr_mean']:.2f}",
                    "ΔR(H)":f"{best.get('sync_gain_mean',0):.3f}"
                })
                pbar.update(1)

    # Kurz-Report
    try:
        df = pd.read_csv(out_path)
        print("\n=== Netzwerk-Highlights (HYP >= 5σ & ANT < 5σ & ΔR>0) ===")
        key=["N","curvature","loss_mm","alpha","ell_um","node_radius_um","kappa_sigma",
             "det_x_um","det_y_um","det_z_um","win_ns"]
        for c in key+["mode","snr_mean","sync_gain_mean"]:
            if c not in df.columns:
                raise RuntimeError(f"Spalte fehlt: '{c}'. Spalten: {list(df.columns)}")
        df["_ord"] = np.arange(len(df))
        sel = (df.sort_values(["snr_mean","_ord"])
                 .groupby(key+["mode"], as_index=False)
                 .tail(1))
        hyp = sel[sel["mode"]=="hypothesis"].set_index(key)
        ant = sel[sel["mode"]=="antithesis"].set_index(key)
        common = hyp.index.intersection(ant.index)

        any_case=False
        for idx in common:
            h, a = hyp.loc[idx], ant.loc[idx]
            if float(h["snr_mean"])>=5.0 and float(a["snr_mean"])<5.0 and float(h["sync_gain_mean"])>0:
                print(f"{idx} -> HYP SNR={h['snr_mean']:.2f}, ANT SNR={a['snr_mean']:.2f}, ΔR={h['sync_gain_mean']:.3f}")
                any_case=True
        if not any_case: print("(keine passenden Fälle gefunden)")
        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")
    except Exception as e:
        print(f"\nHinweis: Zusammenfassung nicht möglich: {e}")
        try:
            print("Spalten:", list(pd.read_csv(out_path, nrows=0).columns))
        except Exception:
            pass
        print(f"Rohdaten: {out_path.resolve()}")

if __name__ == "__main__":
    main()
