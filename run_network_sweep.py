#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network-level sweep für MT/UPE:
- Mehrere MT-Segmente (N Nodes)
- Photonic coupling (exp(-d/ell))
- Hypothese (Superradiance) vs. Antithese (inkoherent)
- PRC-basierter Phase-Reset und Kuramoto-Synchronisation
- Detektor mit Window/Deadtime/Afterpulse
- tqdm-ETA und CSV-Export (Mittelwerte + SEM)

Nutzung (Beispiel):
  venv-mtsim/bin/python run_network_sweep.py --preset sweetspot --N 12 --alpha 0.3 --ells-um 8 \
      --trials 12 --seed 777 --window-ns 100 --deadtime-ns 50 --afterpulse 0.02 \
      --out sweep_network.csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Optional, Callable, Iterable, Tuple

import numpy as np
import pandas as pd

# tqdm (mit Fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Projekt-Module
from modules.photonic import PhotonicEmitter  # Hypothesis=SR, Antithesis=incoherent
from modules.detector import Detector

# ------------------------ Hilfsfunktionen: Geometrie & Netz -------------------

def place_nodes(N: int, radius_um: float, seed: int) -> np.ndarray:
    """Random uniforme Punkte in Kugel (µm)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < N:
        p = rng.uniform(-radius_um, radius_um, size=3)
        if np.linalg.norm(p) <= radius_um:
            pts.append(p)
    return np.array(pts)  # [N,3], in µm

def pairwise_dist_um(P: np.ndarray) -> np.ndarray:
    """P: [N,3] in µm -> Distanzmatrix in µm."""
    diff = P[:, None, :] - P[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, 0.0)
    return D

def coupling_matrix(D_um: np.ndarray, ell_um: float, alpha: float) -> np.ndarray:
    """K_ij = alpha * exp(-d_ij/ell), K_ii=0."""
    if ell_um <= 0:  # entkoppelt
        return np.zeros_like(D_um)
    K = alpha * np.exp(-D_um / ell_um)
    np.fill_diagonal(K, 0.0)
    return K

# ------------------------ PRC & Synchronisation ------------------------------

def prc_kuramoto_delta_phi(phi: np.ndarray, Ipeak_norm: np.ndarray, eps: float = 0.2, phi0: float = 0.0) -> np.ndarray:
    """
    Einfache (type-I) PRC: Δφ = eps * sin(phi - phi0) * I_peak_norm
    - phi: initial phases [N] in rad
    - Ipeak_norm: [0..1] normierte Peaks je Node
    - eps: Koppelstärke des externen Pulses
    """
    return eps * np.sin(phi - phi0) * Ipeak_norm

def kuramoto_order_parameter(phi: np.ndarray) -> float:
    z = np.exp(1j * phi)
    R = np.abs(np.mean(z))
    return float(R)

# ------------------------ Einzellauf Netzwerk --------------------------------

def simulate_network_once(
    N: int,
    curvature: float,
    mode: str,
    loss_mm: float,
    det_pos_um: Tuple[float, float, float],
    node_radius_um: float,
    alpha: float,
    ell_um: float,
    window_s: float,
    deadtime_s: float,
    afterpulse: float,
    seed: int
) -> dict:
    """
    Ein Netzwerk-Run:
    - N MT-Knoten mit PhotonicEmitter (gleiches curvature/mode)
    - Kopplungsmatrix mischt I(t) zwischen Nodes
    - Ein gemeinsamer Detektor (Position in µm)
    - PRC-basiertes Phase-Reset auf lokale Oszillatoren
    - Metriken: det_photons, SNR, R_before/after, sync_gain, mean_phase_reset
    """
    rng = np.random.default_rng(seed)

    # 1) Node-Positionen & Distanzen
    P = place_nodes(N, radius_um=node_radius_um, seed=seed)
    D = pairwise_dist_um(P)
    K = coupling_matrix(D, ell_um=ell_um, alpha=alpha)  # [N,N]

    # 2) Photonik-Zeitachsen je Node (identische Zeitbasis)
    dt = 1e-12
    T = 5e-9
    time = np.arange(0, T, dt)

    # Basissignale je Node
    I_base = []
    for _ in range(N):
        phot = PhotonicEmitter(cluster_size=50, n_clusters=20, mode=mode, curvature=curvature)
        I_base.append(phot.emit(time))
    I_base = np.array(I_base)  # [N, T]

    # 3) Photonische Kopplung: I_net = I_base + K @ I_base  (lineare Mischung, zeitpunktweise)
    # effizient: I_net[n,t] = I_base[n,t] + sum_j K[n,j]*I_base[j,t]
    I_net = I_base + K @ I_base  # Broadcasting über Zeit dank matrix @ array [N,T]

    # 4) Detektion: ein Detektor; Photonen aus jedem Node mit Dämpfung abhängig von Node->Detektor Abstand
    det = Detector(QE=0.6, eta_geom=0.1, mu_eff_mm=loss_mm,
                   r_um=1.0, dark=100, window=window_s)  # r_um wird node-spezifisch überschrieben unten

    # Emittierte Photonen je Node
    hbar = 1.054e-34; c = 3e8; lam = 280e-9; omega = 2*np.pi*c/lam
    E_photon = hbar*omega
    N_emit_nodes = np.trapz(I_net, time, axis=1) / E_photon  # [N]

    # Node->Detektor Dämpfung + Summation der Zählungen
    det_pos = np.array(det_pos_um)  # [3] in µm
    r_nodes_um = np.linalg.norm(P - det_pos[None, :], axis=1)  # [N]
    mu = loss_mm * 1000.0  # 1/m
    r_m = r_nodes_um * 1e-6
    attenuation_nodes = np.exp(-mu * r_m)
    N_det = float(np.sum(N_emit_nodes * attenuation_nodes) * det.eta * det.QE)

    # Deadtime & Afterpulsing
    if window_s > 0 and deadtime_s > 0:
        N_det = N_det / (1.0 + (N_det * deadtime_s / window_s))
    if afterpulse > 0:
        N_det = N_det * (1.0 + afterpulse)

    dark_counts = det.dark * det.win
    snr = N_det / np.sqrt(N_det + dark_counts)

    # 5) PRC-basierter Phase-Reset & Synchronisation
    phi0 = rng.uniform(0, 2*np.pi, size=N)
    R_before = kuramoto_order_parameter(phi0)

    Ipeak = I_net.max(axis=1)  # [N]
    Ipeak_norm = (Ipeak / (Ipeak.max() + 1e-30))
    dphi = prc_kuramoto_delta_phi(phi0, Ipeak_norm, eps=0.2, phi0=0.0)
    phi1 = (phi0 + dphi) % (2*np.pi)
    R_after = kuramoto_order_parameter(phi1)

    sync_gain = R_after - R_before
    mean_phase_reset = float(np.mean(np.abs(dphi)))

    return {
        "mode": mode,
        "N": N,
        "curvature": float(curvature),
        "loss_mm": float(loss_mm),
        "alpha": float(alpha),
        "ell_um": float(ell_um),
        "det_x_um": float(det_pos_um[0]),
        "det_y_um": float(det_pos_um[1]),
        "det_z_um": float(det_pos_um[2]),
        "node_radius_um": float(node_radius_um),
        "emit_photons_total": float(np.sum(N_emit_nodes)),
        "det_photons_total": float(N_det),
        "snr": float(snr),
        "R_before": float(R_before),
        "R_after": float(R_after),
        "sync_gain": float(sync_gain),
        "mean_phase_reset": mean_phase_reset
    }

# ------------------------ Mittelung über Trials ------------------------------

def simulate_network_avg(
    N: int,
    curvature: float,
    mode: str,
    loss_mm: float,
    det_pos_um: Tuple[float, float, float],
    node_radius_um: float,
    alpha: float,
    ell_um: float,
    trials: int,
    base_seed: int,
    window_s: float,
    deadtime_s: float,
    afterpulse: float,
    pbar: Optional[Callable[[Iterable], Iterable]] = None
) -> dict:
    assert trials > 0, "trials muss > 0 sein"
    rows = []
    iterator = range(trials)
    if pbar is not None:
        iterator = pbar(iterator, total=trials, leave=False,
                        desc=f"N={N} {mode}, κ={curvature}, μ={loss_mm}, α={alpha}, ℓ={ell_um}µm")
    for i in iterator:
        seed = base_seed + i
        rows.append(simulate_network_once(
            N, curvature, mode, loss_mm, det_pos_um, node_radius_um,
            alpha, ell_um, window_s, deadtime_s, afterpulse, seed
        ))

    # Meta fest
    meta_keys = {"mode", "N", "curvature", "loss_mm", "alpha", "ell_um",
                 "det_x_um", "det_y_um", "det_z_um", "node_radius_um"}
    out = {"trials": trials}
    for k in meta_keys:
        out[k] = rows[0][k]

    # Mittelwerte + SEM
    for k, v in rows[0].items():
        if k in meta_keys or not isinstance(v, (int, float)):
            continue
        arr = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(arr.mean())
        out[k + "_sem"] = float(arr.std(ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0

    return out

# ------------------------ CSV & Preset --------------------------------------

EXPECTED_FIELDS = [
    "trials", "mode", "N", "curvature", "loss_mm", "alpha", "ell_um",
    "det_x_um", "det_y_um", "det_z_um", "node_radius_um",
    "emit_photons_total_mean", "emit_photons_total_sem",
    "det_photons_total_mean", "det_photons_total_sem",
    "snr_mean", "snr_sem",
    "R_before_mean", "R_before_sem",
    "R_after_mean", "R_after_sem",
    "sync_gain_mean", "sync_gain_sem",
    "mean_phase_reset_mean", "mean_phase_reset_sem"
]

def header_mismatch(cols) -> bool:
    return list(cols) != EXPECTED_FIELDS

def build_grid(args) -> list[Tuple[float, float, float, str]]:
    if args.preset is None or args.preset == "default":
        curvatures = [float(x) for x in args.curvatures.split(",") if x.strip()]
        losses = [float(x) for x in args.losses.split(",") if x.strip()]
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        ells = [float(x) for x in args.ells_um.split(",") if x.strip()]
    elif args.preset == "sweetspot":
        curvatures = [0.8, 0.9, 1.0, 1.1]
        losses = [3, 5, 7, 10, 15]
        alphas = [0.1, 0.2, 0.3, 0.5]
        ells = [5, 8, 12, 20]  # µm
    else:
        raise ValueError(f"Unbekanntes Preset: {args.preset}")

    combos = [(k, mu, a, ell, m)
              for k in curvatures
              for mu in losses
              for a in alphas
              for ell in ells
              for m in ["hypothesis", "antithesis"]]
    return combos

# ------------------------ CLI & Runner --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Network-level MT/UPE Sweep (Hypothese vs. Antithese)")
    ap.add_argument("--preset", type=str, default="default", choices=["default", "sweetspot"])
    ap.add_argument("--N", type=int, default=12, help="Anzahl MT-Segmente im Netzwerk.")
    ap.add_argument("--node-radius-um", type=float, default=12.0, help="Kugelradius für Node-Placement (µm).")
    ap.add_argument("--det-pos-um", type=str, default="0,0,20", help="Detektorposition x,y,z in µm (CSV).")
    ap.add_argument("--curvatures", type=str, default="0.5,1.0", help="κ (wird von --preset überschrieben).")
    ap.add_argument("--losses", type=str, default="5,10,30", help="μ_eff mm^-1 (wird von --preset überschrieben).")
    ap.add_argument("--alphas", type=str, default="0.1,0.3", help="Kopplungsstärke α (wird von --preset überschrieben).")
    ap.add_argument("--ells-um", type=str, default="8,12", help="Kopplungs-Längenskala ℓ in µm (wird von --preset überschrieben).")

    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1000)

    ap.add_argument("--window-ns", type=float, default=100.0)
    ap.add_argument("--deadtime-ns", type=float, default=0.0)
    ap.add_argument("--afterpulse", type=float, default=0.0)

    ap.add_argument("--out", type=str, default="sweep_network.csv")
    ap.add_argument("--no-header", action="store_true")

    args = ap.parse_args()

    det_pos_um = tuple(float(x) for x in args.det_pos_um.split(","))
    window_s = args.window_ns * 1e-9
    deadtime_s = args.deadtime_ns * 1e-9
    afterpulse = float(args.afterpulse)

    out_path = Path(args.out)

    # CSV-Header-Handling
    mode = "a" if out_path.exists() else "w"
    if out_path.exists():
        try:
            cols = list(pd.read_csv(out_path, nrows=0).columns)
            if header_mismatch(cols):
                bak = out_path.with_suffix(out_path.suffix + ".bak")
                out_path.rename(bak)
                print(f"[Info] Header-Mismatch. Backup nach: {bak.name}")
                mode = "w"
        except Exception:
            bak = out_path.with_suffix(out_path.suffix + ".bak")
            out_path.rename(bak)
            print(f"[Info] CSV unlesbar. Backup nach: {bak.name}")
            mode = "w"

    write_header = (mode == "w") and (not args.no_header)

    combos = build_grid(args)
    total = len(combos)

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        if write_header:
            writer.writeheader()

        with tqdm(total=total, desc="Network Sweep", unit="combo") as pbar:
            for (kappa, mu, alpha, ell, mode_lbl) in combos:

                def inner_tqdm(iterable, **kw):
                    return tqdm(iterable, **kw)

                res = simulate_network_avg(
                    N=args.N, curvature=kappa, mode=mode_lbl, loss_mm=mu,
                    det_pos_um=det_pos_um, node_radius_um=args.node_radius_um,
                    alpha=alpha, ell_um=ell, trials=args.trials, base_seed=args.seed,
                    window_s=window_s, deadtime_s=deadtime_s, afterpulse=afterpulse,
                    pbar=inner_tqdm
                )

                row = {k: res.get(k, "") for k in EXPECTED_FIELDS}
                writer.writerow(row); f.flush()

                pbar.set_postfix({
                    "mode": mode_lbl, "κ": kappa, "μ": mu, "α": alpha, "ℓ(µm)": ell,
                    "SNR_mean": f"{res.get('snr_mean', 0.0):.2f}",
                    "R→": f"{res.get('R_before_mean',0.0):.2f}→{res.get('R_after_mean',0.0):.2f}"
                })
                pbar.update(1)

    # kleine Zusammenfassung: Fälle mit ΔR>0 und HYP≥5σ>ANT
    try:
        df = pd.read_csv(out_path)
        print("\n=== Netzwerk-Highlights (HYP >= 5σ & ANT < 5σ & sync_gain_mean > 0) ===")
        # pro Kombination (N, κ, μ, α, ℓ) die Hyp/Ant nebeneinander legen
        key_cols = ["N","curvature","loss_mm","alpha","ell_um"]
        hyp = df[df["mode"]=="hypothesis"].set_index(key_cols)
        ant = df[df["mode"]=="antithesis"].set_index(key_cols)
        common = hyp.index.intersection(ant.index)
        any_case = False
        for idx in common:
            row_h = hyp.loc[idx]
            row_a = ant.loc[idx]
            if (row_h["snr_mean"] >= 5.0) and (row_a["snr_mean"] < 5.0) and (row_h["sync_gain_mean"] > 0):
                N, k, mu, a, ell = idx
                print(f"N={N}, κ={k}, μ={mu} mm^-1, α={a}, ℓ={ell} µm  "
                      f"-> HYP SNR={row_h['snr_mean']:.2f} (ΔR={row_h['sync_gain_mean']:.3f}), "
                      f"ANT SNR={row_a['snr_mean']:.2f}")
                any_case = True
        if not any_case:
            print("(keine passenden Fälle gefunden)")
        print(f"\nErgebnisse gespeichert in: {out_path.resolve()}")
    except Exception as e:
        print(f"\nHinweis: Konnte Netzwerk-Highlights nicht erzeugen: {e}")
        print(f"Rohdaten liegen in {out_path.resolve()}")


if __name__ == "__main__":
    main()
