#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import numpy as np
import pandas as pd


# --------------------------- I/O & Normalisierung ---------------------------

def load_all(csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["__source__"] = Path(p).name
            frames.append(df)
        except Exception as e:
            print(f"[Warn] Konnte Datei nicht laden: {p} ({e})", file=sys.stderr)
    if not frames:
        raise SystemExit("Keine gültigen CSVs geladen.")
    df = pd.concat(frames, ignore_index=True)
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Vereinheitlicht Spaltennamen & füllt fehlende mit Defaults. """
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Einheitliche Typen
    for col in df.columns:
        if df[col].dtype == object:
            # Versuche Zahlen zu casten wo sinnvoll
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    # Häufig benötigte Spalten sicherstellen (falls fehlen -> NaN/Default)
    defaults: Dict[str, float] = dict(
        qe=np.nan, eta_geom=np.nan, dark_cps=np.nan,
        deadtime_ns=np.nan, win_ns=np.nan,
        prc_eps=np.nan, prc_threshold=np.nan,
        N=np.nan, alpha=np.nan, ell_um=np.nan,
        node_radius_um=np.nan, kappa_sigma=np.nan,
        distance_um=np.nan, loss_mm=np.nan, curvature=np.nan,
    )
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # Minimal erforderliche Felder prüfen
    needed = ["mode", "snr_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Erforderliche Spalten fehlen: {missing}")

    return df


# --------------------------- Schlüssel & Paarbildung ------------------------

PARAM_KEYS_DEFAULT = [
    "curvature", "loss_mm", "distance_um", "qe", "eta_geom", "dark_cps",
    "deadtime_ns", "win_ns"
]

NETWORK_KEYS_DEFAULT = [
    "N", "curvature", "loss_mm", "alpha", "ell_um", "node_radius_um",
    "kappa_sigma", "det_x_um", "det_y_um", "det_z_um", "qe", "eta_geom",
    "dark_cps", "deadtime_ns", "win_ns", "prc_eps", "prc_threshold"
]

def auto_keys(df: pd.DataFrame) -> List[str]:
    """ Wähle passende Schlüssel je nach vorhandenen Spalten. """
    # Netzwerk, wenn Alpha/Ell/Detektor-Pos vorhanden
    if {"alpha", "ell_um"}.issubset(df.columns):
        keys = [k for k in NETWORK_KEYS_DEFAULT if k in df.columns]
    else:
        keys = [k for k in PARAM_KEYS_DEFAULT if k in df.columns]
    # Reihenfolge stabilisieren
    return keys


def pair_hyp_ant(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """ Paart Hypothesis vs. Antithesis je Keys; berechnet Differenzen & Ratios. """
    if not keys:
        raise SystemExit("Leere Schlüssel-Liste erhalten.")

    hyp = df[df["mode"] == "hypothesis"].copy()
    ant = df[df["mode"] == "antithesis"].copy()

    # Mehrdeutigkeiten reduzieren: wenn es mehrere Zeilen pro Key gibt (z.B. aus Mehrfachläufen),
    # nimm die letzte oder die mit höchster SNR (hier: höchste SNR).
    def reduce_group(g: pd.DataFrame) -> pd.Series:
        return g.sort_values("snr_mean", kind="mergesort").iloc[-1]

    hyp_red = hyp.groupby(keys, dropna=False, as_index=False).apply(reduce_group).reset_index(drop=True)
    ant_red = ant.groupby(keys, dropna=False, as_index=False).apply(reduce_group).reset_index(drop=True)

    # Merge
    suffixes = ("_H", "_A")
    pairs = pd.merge(hyp_red, ant_red, on=keys, how="inner", suffixes=suffixes)
    if pairs.empty:
        raise SystemExit("Keine HYP/ANT-Paare mit gemeinsamen Schlüsseln gefunden.")

    # Kennzahlen
    pairs["snr_gap"] = pairs["snr_mean_H"] - pairs["snr_mean_A"]
    pairs["snr_ratio"] = pairs["snr_mean_H"] / (pairs["snr_mean_A"] + 1e-30)

    # Optional vorhandene Qualitätsmetriken
    for m in ["coincidence_ratio_mean", "p2rms_mean", "peak_sharpness_mean",
              "gate_photons_mean", "det_photons_total_mean"]:
        h = m + "_H"; a = m + "_A"
        if m + "_H" not in pairs.columns and m + "_A" not in pairs.columns:
            if m in hyp_red.columns and m in ant_red.columns:
                # diese sind nicht automatisch mit suffixen entstanden (wegen groupby-reduce)
                # also mergen wir sie nachträglich:
                pairs[h] = pairs[m + "_x"] if m + "_x" in pairs.columns else np.nan
                pairs[a] = pairs[m + "_y"] if m + "_y" in pairs.columns else np.nan
            continue
        # Gaps & Ratios nur, wenn beide da sind
        if h in pairs.columns and a in pairs.columns:
            pairs[m + "_gap"] = pairs[h] - pairs[a]
            pairs[m + "_ratio"] = pairs[h] / (pairs[a] + 1e-30)

    return pairs


# ------------------------------ Berichte ------------------------------------

def discrimination_table(pairs: pd.DataFrame,
                         min_hyp_snr: float,
                         max_ant_snr: float) -> pd.DataFrame:
    sel = (pairs["snr_mean_H"] >= min_hyp_snr) & (pairs["snr_mean_A"] < max_ant_snr)
    return pairs.loc[sel].copy().sort_values(["snr_mean_H", "snr_gap"], ascending=[False, False])


def top_table(pairs: pd.DataFrame, sort_by: str, topk: int) -> pd.DataFrame:
    sort_by = sort_by if sort_by in pairs.columns else "snr_gap"
    return pairs.sort_values(sort_by, ascending=False).head(topk).copy()


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "pairs_total": [len(pairs)],
        "snr_gap_mean": [pairs["snr_gap"].mean()],
        "snr_gap_median": [pairs["snr_gap"].median()],
        "snr_H_mean": [pairs["snr_mean_H"].mean()],
        "snr_A_mean": [pairs["snr_mean_A"].mean()],
    })


# ------------------------------- CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyse-Skript für MT/UPE Sweeps (Param & Network).")
    ap.add_argument("--csv", type=str, nargs="+", required=True, help="Eine oder mehrere CSV-Dateien.")
    ap.add_argument("--keys", type=str, default="",
                    help="Kommaliste zur expliziten Key-Wahl (überschreibt Auto-Keys). "
                         "Beispiel Param: 'curvature,loss_mm,distance_um,qe,eta_geom,dark_cps,deadtime_ns,win_ns' "
                         "Beispiel Network: 'N,curvature,loss_mm,alpha,ell_um,node_radius_um,kappa_sigma,det_x_um,det_y_um,det_z_um,qe,eta_geom,dark_cps,deadtime_ns,win_ns,prc_eps,prc_threshold'")
    ap.add_argument("--min-hyp-snr", type=float, default=5.0, help="Schwelle für Hypothesis.")
    ap.add_argument("--max-ant-snr", type=float, default=5.0, help="Schwelle für Antithesis.")
    ap.add_argument("--sort-by", type=str, default="snr_gap", help="Sortierschlüssel für Top-Tabelle.")
    ap.add_argument("--topk", type=int, default=25, help="Top-N Zeilen in der Übersicht.")
    ap.add_argument("--out-prefix", type=str, default="analysis", help="Präfix für Output-CSVs.")
    args = ap.parse_args()

    df = load_all(args.csv)

    # Keys bestimmen/ersetzen
    if args.keys.strip():
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    else:
        keys = auto_keys(df)

    print(f"[Info] Verwende Keys: {keys}")

    pairs = pair_hyp_ant(df, keys)

    disc = discrimination_table(pairs, args.min_hyp_snr, args.max_ant_snr)
    top = top_table(pairs, args.sort_by, args.topk)
    summ = summarize(pairs)

    # Ausgaben
    out_prefix = Path(args.out_prefix)
    out_all = out_prefix.with_suffix(".pairs.csv")
    out_disc = out_prefix.with_suffix(".discriminations.csv")
    out_top = out_prefix.with_suffix(".top.csv")
    out_summary = out_prefix.with_suffix(".summary.csv")

    pairs.to_csv(out_all, index=False)
    disc.to_csv(out_disc, index=False)
    top.to_csv(out_top, index=False)
    summ.to_csv(out_summary, index=False)

    # Konsole: kompakte Übersicht
    print("\n=== Summary ===")
    print(summ.to_string(index=False))

    print("\n=== Top nach", args.sort_by, f"(Top {args.topk}) ===")
    cols_print = ["snr_mean_H", "snr_mean_A", "snr_gap", "snr_ratio"]
    cols_print = [c for c in cols_print if c in top.columns]
    # plus Schlüsselspalten
    cols_print = keys + cols_print
    print(top[cols_print].to_string(index=False))

    print("\n=== Discriminations (HYP >= %.2f & ANT < %.2f) ===" % (args.min_hyp_snr, args.max_ant_snr))
    if disc.empty:
        print("(keine passenden Fälle gefunden)")
    else:
        cols_d = keys + [c for c in ["snr_mean_H", "snr_mean_A", "snr_gap", "snr_ratio"] if c in disc.columns]
        print(disc[cols_d].head(args.topk).to_string(index=False))

    print(f"\n[OK] Dateien geschrieben:\n  {out_all}\n  {out_disc}\n  {out_top}\n  {out_summary}")


if __name__ == "__main__":
    main()
