#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv
from pathlib import Path
import pandas as pd
from mt_upe_core import simulate_network_avg

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

EXPECTED_FIELDS = [
    "trials","mode","N","curvature","loss_mm","alpha","ell_um",
    "det_x_um","det_y_um","det_z_um","node_radius_um","kappa_sigma",
    "qe","eta_geom","dark_cps","deadtime_ns","win_ns","prc_eps","prc_threshold",
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

def parse_list_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]

def build_grid(args):
    if args.preset == "sweetspot":
        curvatures=[0.8,0.9,1.0,1.1]
        losses=[3,5,7,10,15]
        alphas=[0.1,0.2,0.3,0.5,0.8,1.0]
        ells=[5,8,12,20]
    else:
        curvatures=parse_list_floats(args.curvatures)
        losses=parse_list_floats(args.losses)
        alphas=parse_list_floats(args.alphas)
        ells=parse_list_floats(args.ells_um)
    return [(k,mu,a,ell) for k in curvatures for mu in losses for a in alphas for ell in ells]

def main():
    ap = argparse.ArgumentParser(description="Network-level MT/UPE Sweep (HYP vs ANT, PRC, Sync)")
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

    ap.add_argument("--window-ns", type=float, default=100.0)
    ap.add_argument("--optimize-window", type=str, default="",
                    help="Kommagetrennte Liste ns (z.B. '0.1,0.2,0.5'); wählt bestes Gate per HYP und nutzt es auch für ANT.")
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)
    ap.add_argument("--qe", type=float, default=0.6)
    ap.add_argument("--eta-geom", type=float, default=0.1)
    ap.add_argument("--dark-cps", type=float, default=100.0)
    ap.add_argument("--kappa-sigma", type=float, default=0.0)
    ap.add_argument("--emission-scale", type=float, default=1.0)

    # prc-eps als Kommaliste
    ap.add_argument("--prc-eps", type=str, default="0.2",
                    help="Kommaliste; PRC-Kopplungsstärke(n): Δφ = eps * sin(φ - φ0) * Ipeak_norm")
    ap.add_argument("--prc-threshold", type=float, default=0.0,
                    help="Minimale Peak-to-RMS der Detektor-Summenwelle, ab der PRC angewendet wird (0=immer).")

    ap.add_argument("--out", type=str, default="sweep_network.csv")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    det_pos = tuple(float(x) for x in args.det_pos_um.split(","))
    ds = args.deadtime_ns*1e-9

    win_list = [args.window_ns] if not args.optimize_window else \
               [float(x) for x in args.optimize_window.split(",") if x.strip()]

    prc_list = parse_list_floats(args.prc_eps)

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
    total = len(grid) * max(1, len(prc_list))

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        if write_header: writer.writeheader()

        with tqdm(total=total, desc="Network Sweep", unit="combo") as pbar:
            for (kappa, mu, alpha, ell) in grid:
                def inner_tqdm(it, **kw): return tqdm(it, **kw)

                for prc_eps in prc_list:
                    # 1) Hypothese: bestes Gate suchen
                    best = None
                    for wn in win_list:
                        res_try = simulate_network_avg(
                            N=args.N, curvature=kappa, mode="hypothesis", loss_mm=mu,
                            det_pos_um=det_pos, node_radius_um=args.node_radius_um,
                            alpha=alpha, ell_um=ell, trials=args.trials, base_seed=args.seed,
                            window_s=wn*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                            qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                            kappa_sigma=args.kappa_sigma, emission_scale=args.emission_scale,
                            pbar=inner_tqdm, prc_eps=prc_eps, prc_threshold=args.prc_threshold
                        )
                        if (best is None) or (res_try["snr_mean"] > best["snr_mean"]):
                            best = res_try; best["win_ns"] = wn

                    print(f"[HYP] N={args.N}, κ={kappa}, μ={mu}, α={alpha}, ℓ={ell} µm, "
                          f"win={best['win_ns']} ns, prc_eps={prc_eps} | "
                          f"SNR={best['snr_mean']:.2f}, ΔR={best.get('sync_gain_mean',0.0):.3f}, "
                          f"CR={best.get('coincidence_ratio_mean',0.0):.3f}, p2rms={best.get('p2rms_mean',0.0):.2f}, "
                          f"gateN={best.get('gate_photons_mean',0.0):.3f}")

                    row_h = {k: "" for k in EXPECTED_FIELDS}
                    row_h.update({
                        "trials": args.trials, "mode": "hypothesis",
                        "N": args.N, "curvature": kappa, "loss_mm": mu, "alpha": alpha, "ell_um": ell,
                        "det_x_um": det_pos[0], "det_y_um": det_pos[1], "det_z_um": det_pos[2],
                        "node_radius_um": args.node_radius_um, "kappa_sigma": args.kappa_sigma,
                        "qe": args.qe, "eta_geom": args.eta_geom, "dark_cps": args.dark_cps,
                        "deadtime_ns": args.deadtime_ns, "win_ns": best["win_ns"],
                        "prc_eps": prc_eps, "prc_threshold": args.prc_threshold
                    })
                    for k,v in best.items():
                        if k+"_mean" in EXPECTED_FIELDS: row_h[k+"_mean"] = v
                        if k+"_sem"  in EXPECTED_FIELDS: row_h[k+"_sem"]  = v
                    writer.writerow(row_h); f.flush()

                    # 2) Antithese: identisches Gate
                    res_ant = simulate_network_avg(
                        N=args.N, curvature=kappa, mode="antithesis", loss_mm=mu,
                        det_pos_um=det_pos, node_radius_um=args.node_radius_um,
                        alpha=alpha, ell_um=ell, trials=args.trials, base_seed=args.seed,
                        window_s=best["win_ns"]*1e-9, deadtime_s=ds, afterpulse=args.afterpulse,
                        qe=args.qe, eta_geom=args.eta_geom, dark_cps=args.dark_cps,
                        kappa_sigma=args.kappa_sigma, emission_scale=args.emission_scale,
                        pbar=inner_tqdm, prc_eps=prc_eps, prc_threshold=args.prc_threshold
                    )

                    print(f"[ANT] N={args.N}, κ={kappa}, μ={mu}, α={alpha}, ℓ={ell} µm, "
                          f"win={best['win_ns']} ns, prc_eps={prc_eps} | "
                          f"SNR={res_ant['snr_mean']:.2f}, ΔR={res_ant.get('sync_gain_mean',0.0):.3f}, "
                          f"CR={res_ant.get('coincidence_ratio_mean',0.0):.3f}, p2rms={res_ant.get('p2rms_mean',0.0):.2f}, "
                          f"gateN={res_ant.get('gate_photons_mean',0.0):.3f}")

                    row_a = {k: "" for k in EXPECTED_FIELDS}
                    row_a.update({
                        "trials": args.trials, "mode": "antithesis",
                        "N": args.N, "curvature": kappa, "loss_mm": mu, "alpha": alpha, "ell_um": ell,
                        "det_x_um": det_pos[0], "det_y_um": det_pos[1], "det_z_um": det_pos[2],
                        "node_radius_um": args.node_radius_um, "kappa_sigma": args.kappa_sigma,
                        "qe": args.qe, "eta_geom": args.eta_geom, "dark_cps": args.dark_cps,
                        "deadtime_ns": args.deadtime_ns, "win_ns": best["win_ns"],
                        "prc_eps": prc_eps, "prc_threshold": args.prc_threshold
                    })
                    for k,v in res_ant.items():
                        if k+"_mean" in EXPECTED_FIELDS: row_a[k+"_mean"] = v
                        if k+"_sem"  in EXPECTED_FIELDS: row_a[k+"_sem"]  = v
                    writer.writerow(row_a); f.flush()

                    pbar.set_postfix({
                        "κ": kappa, "μ": mu, "α": alpha, "ℓ(µm)": ell,
                        "win(ns)": best["win_ns"], "prc": prc_eps,
                        "SNR(H)": f"{best.get('snr_mean',0.0):.2f}",
                        "SNR(A)": f"{res_ant.get('snr_mean',0.0):.2f}",
                        "ΔR(H)": f"{best.get('sync_gain_mean',0.0):.3f}"
                    })
                    pbar.update(1)

    # Kurz-Report
    try:
        df = pd.read_csv(out_path)
        print("\n=== Netzwerk-Highlights (HYP >= 5σ & ANT < 5σ & ΔR>0) ===")
        key=["N","curvature","loss_mm","alpha","ell_um","node_radius_um","kappa_sigma",
             "det_x_um","det_y_um","det_z_um","qe","eta_geom","dark_cps","deadtime_ns","win_ns","prc_eps","prc_threshold"]
        hyp = df[df["mode"]=="hypothesis"].set_index(key)
        ant = df[df["mode"]=="antithesis"].set_index(key)
        common = hyp.index.intersection(ant.index)
        any_case=False
        for idx in common:
            h, a = hyp.loc[idx], ant.loc[idx]
            if h["snr_mean"]>=5.0 and a["snr_mean"]<5.0 and h["sync_gain_mean"]>0:
                print(f"{idx} -> HYP SNR={h['snr_mean']:.2f}, ANT SNR={a['snr_mean']:.2f}, ΔR={h['sync_gain_mean']:.3f}")
                any_case=True
        if not any_case: print("(keine passenden Fälle gefunden)")
        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")
    except Exception as e:
        print(f"\nHinweis: Zusammenfassung nicht möglich: {e}")
        print(f"Rohdaten: {out_path.resolve()}")

if __name__ == "__main__":
    main()
