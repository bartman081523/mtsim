#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Optional, Callable, Iterable

import numpy as np
import pandas as pd

# tqdm (mit Fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # Fallback ohne ETA
        return x

# ---- Projekt-Module ---------------------------------------------------------
from adapters.mt_spindle import MTSpindleSystem
from adapters.kmc_motors import MotorLattice
from modules.photonic import PhotonicEmitter
from modules.spin_ros import SpinROS
from modules.detector import Detector


# ------------------------------ Kernfunktionen -------------------------------

def simulate_once(curvature: float, mode: str, loss_mm: float, r_um: float,
                  seed: int, window_s: float, deadtime_s: float, afterpulse: float) -> dict:
    """Einzellauf (ein Trial) mit fixen Meta-Parametern und erweitertem Detektor-Modell."""
    np.random.seed(seed)

    # 1) Zellsysteme (MT-Spindel + Motoren)
    mt = MTSpindleSystem(n_mt=9, length_um=3.0, seed=seed)
    motors = MotorLattice(mt_graph=getattr(mt, "graph", {"n": 9}))

    # 2) Photonik/Spin
    phot = PhotonicEmitter(cluster_size=50, n_clusters=20, mode=mode, curvature=curvature)
    spin = SpinROS(curvature=curvature)  # ROS-Modulator

    # 3) Zeitachse
    dt = 1e-12
    T = 5e-9
    time = np.arange(0, T, dt)

    # 4) Photonik (UPE)
    I_t = phot.emit(time)

    # 5) Motor-Kopplung
    motors.step_series(I_t)

    # 6) Spindel (4-State) mit ROS-Modulator
    mt.update_dynamics(dt_series_s=time * 0 + dt, ros_modulator=spin.modulator())

    # 7) Detektion/SNR (mit Window, Deadtime, Afterpulsing)
    det = Detector(QE=0.6, eta_geom=0.1, mu_eff_mm=loss_mm, r_um=r_um,
                   dark=100, window=window_s)
    N_emit = phot.total_photons(I_t, time)
    N_det, snr = det.measure(N_emit)

    # Deadtime-Korrektur (nicht-paralysierbares Modell, Näherung über Raten)
    # N' = N / (1 + N * tau_d / window)
    if window_s > 0 and deadtime_s > 0:
        N_det = N_det / (1.0 + (N_det * deadtime_s / window_s))

    # Afterpulsing (einfacher Multiplikator)
    if afterpulse > 0:
        N_det = N_det * (1.0 + afterpulse)

    # SNR neu mit korrigierten Zählungen berechnen
    dark_counts = det.dark * det.win
    snr = N_det / np.sqrt(N_det + dark_counts)

    return {
        "mode": mode,
        "curvature": float(curvature),
        "loss_mm": float(loss_mm),
        "distance_um": float(r_um),
        "emit_photons": float(N_emit),
        "det_photons": float(N_det),
        "snr": float(snr),
        "motor_net_steps": int(motors.net_displacement()),
        "photon_peak": float(I_t.max()),
        "cilia_phase_reset": float(phot.phase_reset(I_t)),
        "spindle_metric": float(mt.spindle_metric(spin.modulator()))
    }


def simulate_avg(curvature: float, mode: str, loss_mm: float, r_um: float, trials: int, base_seed: int,
                 window_s: float, deadtime_s: float, afterpulse: float,
                 pbar: Optional[Callable[[Iterable], Iterable]] = None) -> dict:
    """Mittelung über mehrere Trials – Meta-Parameter bleiben **fix** (kein *_mean)."""
    assert trials > 0, "trials muss > 0 sein"

    rows = []
    iterator = range(trials)
    if pbar is not None:
        iterator = pbar(iterator, total=trials, leave=False,
                        desc=f"Trials {mode}, κ={curvature}, μ={loss_mm}, r={r_um}µm")

    for i in iterator:
        seed = base_seed + i
        row = simulate_once(curvature, mode, loss_mm, r_um, seed,
                            window_s=window_s, deadtime_s=deadtime_s, afterpulse=afterpulse)
        rows.append(row)

    # Meta-Schlüssel fix halten
    meta_keys = {"mode", "curvature", "loss_mm", "distance_um"}
    out = {"trials": trials}
    out["mode"] = rows[0]["mode"]
    out["curvature"] = rows[0]["curvature"]
    out["loss_mm"] = rows[0]["loss_mm"]
    out["distance_um"] = rows[0]["distance_um"]

    # Numerische Ergebnis-Keys mitteln (ohne Meta)
    for k, v in rows[0].items():
        if k in meta_keys or not isinstance(v, (int, float)):
            continue
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_sem"] = float(vals.std(ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0

    return out


# ------------------------------ CSV Utilities --------------------------------

EXPECTED_FIELDS = [
    "trials", "mode", "curvature", "loss_mm", "distance_um",
    "emit_photons_mean", "emit_photons_sem",
    "det_photons_mean", "det_photons_sem",
    "snr_mean", "snr_sem",
    "motor_net_steps_mean", "motor_net_steps_sem",
    "photon_peak_mean", "photon_peak_sem",
    "cilia_phase_reset_mean", "cilia_phase_reset_sem",
    "spindle_metric_mean", "spindle_metric_sem"
]

def header_mismatch(existing_cols) -> bool:
    return list(existing_cols) != EXPECTED_FIELDS


# ------------------------------ CLI & Runner --------------------------------

def build_grid(args):
    """Erzeugt Parameter-Kombinationen – entweder manuell oder via Preset."""
    if args.preset is None or args.preset == "default":
        curvatures = [float(x) for x in args.curvatures.split(",") if x.strip() != ""]
        losses = [float(x) for x in args.losses.split(",") if x.strip() != ""]
        distances = [float(x) for x in args.distances.split(",") if x.strip() != ""]
    elif args.preset == "sweetspot":
        # Feinere Suche um die guten Bereiche:
        curvatures = [0.8, 0.9, 1.0, 1.1]
        losses = [3, 5, 7, 10, 15]         # mm^-1
        distances = [5, 10, 20, 35, 50]    # µm
    else:
        raise ValueError(f"Unbekanntes Preset: {args.preset}")

    combos = [(k, mu, r, m)
              for k in curvatures
              for mu in losses
              for r in distances
              for m in ["hypothesis", "antithesis"]]
    return combos


def main():
    ap = argparse.ArgumentParser(description="Parameter-Sweep für MT/UPE Hybrid-Simulation (mit tqdm-ETA).")
    # Manuelle Gitter
    ap.add_argument("--curvatures", type=str, default="0.0,0.5,1.0",
                    help="Kommagetrennt: κ (z.B. '0.0,0.5,1.0'). Wird von --preset überschrieben.")
    ap.add_argument("--losses", type=str, default="5,10,30,100",
                    help="Kommagetrennt: μ_eff in mm^-1. Wird von --preset überschrieben.")
    ap.add_argument("--distances", type=str, default="10,50,100",
                    help="Kommagetrennt: r in µm. Wird von --preset überschrieben.")
    ap.add_argument("--preset", type=str, default="default",
                    choices=["default", "sweetspot"],
                    help="Vordefiniertes Gitter. 'sweetspot' sucht fein in den guten Regionen.")
    # Trials & Seeding
    ap.add_argument("--trials", type=int, default=16, help="Trials pro Kombination.")
    ap.add_argument("--seed", type=int, default=1000, help="Basis-Seed.")
    # Detektor/Timing
    ap.add_argument("--window-ns", type=float, default=100.0, help="Detektor-Gate in ns.")
    ap.add_argument("--deadtime-ns", type=float, default=0.0, help="Nicht-paralysierbare Totzeit in ns.")
    ap.add_argument("--afterpulse", type=float, default=0.0, help="Afterpulsing-Anteil (z.B. 0.02 für 2%).")
    # Output
    ap.add_argument("--out", type=str, default="param_sweep_results.csv", help="CSV-Datei.")
    ap.add_argument("--no-header", action="store_true", help="CSV ohne Header schreiben (append-freundlich).")
    args = ap.parse_args()

    window_s = args.window_ns * 1e-9
    deadtime_s = args.deadtime_ns * 1e-9
    afterpulse = float(args.afterpulse)

    out_path = Path(args.out)

    # CSV: Header-Konsistenz prüfen
    write_mode = "a" if out_path.exists() else "w"
    if out_path.exists():
        try:
            existing_cols = list(pd.read_csv(out_path, nrows=0).columns)
            if header_mismatch(existing_cols):
                bak = out_path.with_suffix(out_path.suffix + ".bak")
                out_path.rename(bak)
                print(f"[Info] Header-Mismatch. Backup nach: {bak.name}")
                write_mode = "w"
        except Exception:
            bak = out_path.with_suffix(out_path.suffix + ".bak")
            out_path.rename(bak)
            print(f"[Info] CSV unlesbar. Backup nach: {bak.name}")
            write_mode = "w"

    write_header = (write_mode == "w") and (not args.no_header)

    combos = build_grid(args)
    total_jobs = len(combos)

    with out_path.open(write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        if write_header:
            writer.writeheader()

        with tqdm(total=total_jobs, desc="Sweep", unit="combo") as pbar_outer:
            for (curv, mu, r, mode) in combos:

                # inneres tqdm für Trials
                def inner_tqdm(iterable, **kw):
                    return tqdm(iterable, **kw)

                res = simulate_avg(curv, mode, mu, r, trials=args.trials, base_seed=args.seed,
                                   window_s=window_s, deadtime_s=deadtime_s, afterpulse=afterpulse,
                                   pbar=inner_tqdm)

                # vollständige Zeile erzwingen
                row = {k: res.get(k, "") for k in EXPECTED_FIELDS}
                writer.writerow(row)
                f.flush()

                pbar_outer.set_postfix({
                    "mode": mode, "κ": curv, "μ(mm^-1)": mu, "r(µm)": r,
                    "SNR_mean": f"{res.get('snr_mean', 0.0):.2f}"
                })
                pbar_outer.update(1)

    # Nach dem Sweep: Diskriminations-Fälle
    try:
        df = pd.read_csv(out_path)

        COL_MODE, COL_CURV, COL_LOSS, COL_DIST = "mode", "curvature", "loss_mm", "distance_um"
        COL_SNR = "snr_mean"

        print("\n=== DISKRIMINATIONS-FÄLLE (HYP >= 5σ & ANT < 5σ) ===")
        curv_vals = sorted(df[COL_CURV].dropna().unique().tolist())
        loss_vals = sorted(df[COL_LOSS].dropna().unique().tolist())
        dist_vals = sorted(df[COL_DIST].dropna().unique().tolist())

        found_any = False
        for curv in curv_vals:
            for mu in loss_vals:
                for r in dist_vals:
                    hyp = df[(df[COL_MODE] == "hypothesis") &
                             (df[COL_CURV] == curv) &
                             (df[COL_LOSS] == mu) &
                             (df[COL_DIST] == r)]
                    ant = df[(df[COL_MODE] == "antithesis") &
                             (df[COL_CURV] == curv) &
                             (df[COL_LOSS] == mu) &
                             (df[COL_DIST] == r)]
                    if len(hyp) == 1 and len(ant) == 1:
                        snr_h = float(hyp[COL_SNR].iloc[0])
                        snr_a = float(ant[COL_SNR].iloc[0])
                        if snr_h >= 5.0 and snr_a < 5.0:
                            print(f"κ={curv}, μ={mu} mm^-1, r={r} µm  -->  HYP SNR={snr_h:.2f},  ANT SNR={snr_a:.2f}")
                            found_any = True
        if not found_any:
            print("(keine Fälle mit HYP ≥5σ und ANT <5σ gefunden)")

        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")

    except Exception as e:
        try:
            print(f"\nFehler in der Auswertung: {e}")
            print("Vorhandene Spalten in CSV:", list(pd.read_csv(out_path, nrows=0).columns))
        except Exception:
            pass
        print(f"Rohdaten liegen in {out_path.resolve()}")


if __name__ == "__main__":
    main()
