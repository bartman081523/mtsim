#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 Decision Analysis:
- Liest eine oder mehrere CSVs (Detailzeilen, HYP/ANT gemischt)
- Normalisiert Modi (hypothesis/antithesis)
- Extrahiert Metriken: score, ΔR (sync_gain)
- Berechnet ROC & AUC (mit 95%-CI via Bootstrap)
- Kombimetrik: z(score)+z(ΔR)
- Sucht optimale Schwelle (Youden J)
- Speichert ROC-Kurven und Summary als CSV; optional PNG-Plots

Beispiel:
  ./venv-mtsim/bin/python stage4_decision_analysis.py \
    --csv stage3_report_clean.csv stage3_report_poisson.csv \
    --out-prefix stage4_clean_poisson --plots
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ------------------------- Helpers -------------------------

def _col_exists(df: pd.DataFrame, names: List[str]) -> str:
    """Finde erste existierende Spalte (case-insensitiv, Unicode-Varianten)."""
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        key = n.lower()
        if key in lower_map:
            return lower_map[key]
    # fallback: versuche startswith
    for c in df.columns:
        cl = c.lower()
        if any(cl.startswith(n.lower()) for n in names):
            return c
    raise KeyError(f"Keine der Spalten gefunden: {names}  (vorhanden: {list(df.columns)[:10]} ...)")


def _norm_mode(val: str) -> str:
    if not isinstance(val, str):
        return str(val)
    v = val.strip().lower()
    if v in ("hypothesis", "hyp", "h"):
        return "hypothesis"
    if v in ("antithesis", "ant", "a"):
        return "antithesis"
    return v


def _prepare(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Gibt (y, groups) zurück (y=1 für HYP, 0 für ANT; groups optional seed/block)."""
    if "mode" not in df.columns:
        raise KeyError("Spalte 'mode' wird benötigt (hypothesis/antithesis).")
    mode = df["mode"].map(_norm_mode)
    y = (mode == "hypothesis").astype(int)

    # Optionale Gruppierung (seed, block o.ä.) – wird hier nicht zwingend genutzt
    groups = None
    for gcol in ("seed", "block", "seed_id"):
        if gcol in df.columns:
            groups = df[gcol]
            break
    return y, groups


def _metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Finden der besten Spaltennamen für:
      - score          (z.B. 'score' oder 'score_mean')
      - sync_gain/ΔR   (z.B. 'deltaR','ΔR','sync_gain','sync_gain_mean')
    """
    cols = {}
    cols["score"] = _col_exists(df, ["score", "score_mean"])
    try:
        cols["sync_gain"] = _col_exists(df, ["Δr", "deltaR", "sync_gain", "sync_gain_mean", "dr", "delta_r"])
    except KeyError:
        # Notfalls aus R_after - R_before rekonstruieren
        if ("R_after" in df.columns) and ("R_before" in df.columns):
            df["_sync_gain_recon"] = df["R_after"] - df["R_before"]
            cols["sync_gain"] = "_sync_gain_recon"
        else:
            raise
    return cols


def _roc_points(y: np.ndarray, s: np.ndarray, direction: str = "higher") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC durch Schwellen-Sweep.
    direction='higher' bedeutet: HYP, wenn s >= thresh.
    Liefert (fpr, tpr, thresh).
    """
    assert y.shape == s.shape
    # Ein bisschen Rauschen für stabile Threshold-Sweeps bei vielen Ties
    s = s.astype(float)
    # Schwellen: eindeutige Werte sortiert
    uniq = np.unique(s)
    if direction == "higher":
        thresholds = np.r_[np.inf, uniq[::-1], -np.inf]
    else:
        thresholds = np.r_[ -np.inf, uniq, np.inf ]

    tpr_list, fpr_list = [], []
    for th in thresholds:
        if direction == "higher":
            yhat = (s >= th).astype(int)
        else:
            yhat = (s <= th).astype(int)
        tp = np.sum((yhat == 1) & (y == 1))
        fp = np.sum((yhat == 1) & (y == 0))
        tn = np.sum((yhat == 0) & (y == 0))
        fn = np.sum((yhat == 0) & (y == 1))
        tpr = tp / (tp + fn + 1e-30)
        fpr = fp / (fp + tn + 1e-30)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # AUC als Fläche unter ROC (FPR aufsteigend sortieren)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, direction: str, n_boot: int = 1000, seed: int = 1234) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        fpr, tpr, _ = _roc_points(yb, sb, direction=direction)
        aucs.append(_auc(fpr, tpr))
    aucs = np.array(aucs, float)
    mean = float(np.mean(aucs))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _youden_threshold(y: np.ndarray, s: np.ndarray, direction: str) -> Tuple[float, float, float, float]:
    """Finde Schwelle mit maximalem Youden-J (TPR-FPR). Gibt (th, tpr, fpr, J)."""
    fpr, tpr, th = _roc_points(y, s, direction=direction)
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    return float(th[j_idx]), float(tpr[j_idx]), float(fpr[j_idx]), float(J[j_idx])


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return (x - m) / s


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 4 Decision Analysis (ROC/AUC/Thresholds) für score, ΔR und Kombination.")
    ap.add_argument("--csv", nargs="+", required=True, help="Eine oder mehrere CSV-Dateien mit HYP/ANT Detailzeilen.")
    ap.add_argument("--out-prefix", type=str, default="stage4_analysis", help="Präfix für Output-Dateien (CSV/PNG).")
    ap.add_argument("--plots", action="store_true", help="PNG-Plots der ROC-Kurven erzeugen.")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Anz. Bootstrap-Resamples für AUC-CI.")
    args = ap.parse_args()

    # Einlesen & vereinigen
    dfs = []
    for p in args.csv:
        try:
            dfp = pd.read_csv(p)
            dfs.append(dfp)
        except Exception as e:
            raise SystemExit(f"[Error] Konnte CSV nicht lesen: {p} ({e})")
    df = pd.concat(dfs, ignore_index=True)

    # Modi & Metriken
    y, _groups = _prepare(df)
    cols = _metric_columns(df)
    score_col = cols["score"]
    sync_col  = cols["sync_gain"]

    s_score = df[score_col].to_numpy(dtype=float)
    s_sync  = df[sync_col].to_numpy(dtype=float)

    # Kombi: z(score) + z(ΔR) (zweiseitig, Richtung 'higher is HYP')
    combo = _zscore(s_score) + _zscore(s_sync)

    # Analyse-Funktion
    def analyze_metric(name: str, s: np.ndarray, direction: str = "higher") -> Dict[str, float]:
        fpr, tpr, _ = _roc_points(y, s, direction=direction)
        auc = _auc(fpr, tpr)
        auc_b, auc_lo, auc_hi = _bootstrap_auc(y, s, direction, n_boot=args.bootstrap)
        th, tpr_opt, fpr_opt, J = _youden_threshold(y, s, direction=direction)
        # Speichere ROC
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_path = f"{args.out_prefix}.roc_{name}.csv"
        roc_df.to_csv(roc_path, index=False)
        return {
            "name": name,
            "auc": auc,
            "auc_boot_mean": auc_b,
            "auc_ci_lo": auc_lo,
            "auc_ci_hi": auc_hi,
            "threshold": th,
            "tpr_at_thr": tpr_opt,
            "fpr_at_thr": fpr_opt,
            "youden_J": J,
            "roc_csv": roc_path
        }

    res_score = analyze_metric("score", s_score, direction="higher")
    res_sync  = analyze_metric("deltaR", s_sync,  direction="higher")
    res_combo = analyze_metric("combo", combo,   direction="higher")

    summary = pd.DataFrame([res_score, res_sync, res_combo])
    sum_path = f"{args.out_prefix}.summary.csv"
    summary.to_csv(sum_path, index=False)

    print("\n=== Stage 4 Decision Analysis ===")
    for r in [res_score, res_sync, res_combo]:
        print(f"[{r['name']}]  AUC={r['auc']:.3f}  (boot {r['auc_boot_mean']:.3f}, 95% CI {r['auc_ci_lo']:.3f}..{r['auc_ci_hi']:.3f})  "
              f"| thr={r['threshold']:.4g}  | TPR={r['tpr_at_thr']:.3f}  | FPR={r['fpr_at_thr']:.3f}  | J={r['youden_J']:.3f}  -> {r['roc_csv']}")
    print(f"\n[OK] Summary -> {sum_path}")

    # Optional: Plots
    if args.plots:
        if not HAVE_PLT:
            print("[Info] matplotlib nicht verfügbar; überspringe Plots.")
        else:
            def plot_roc(roc_csv: str, title: str, png_name: str):
                roc = pd.read_csv(roc_csv)
                plt.figure()
                plt.plot(roc["fpr"], roc["tpr"], lw=2)
                plt.plot([0,1], [0,1], linestyle="--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(png_name, dpi=140)
                plt.close()

            plot_roc(res_score["roc_csv"], "ROC – Score", f"{args.out_prefix}.roc_score.png")
            plot_roc(res_sync["roc_csv"],  "ROC – ΔR",    f"{args.out_prefix}.roc_deltaR.png")
            plot_roc(res_combo["roc_csv"], "ROC – Combo", f"{args.out_prefix}.roc_combo.png")
            print(f"[OK] Plots -> {args.out_prefix}.roc_*.png")


if __name__ == "__main__":
    main()
