#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_final_plots.py

Erzeugt alle finalen, datengestützten Plots für das Manuskript
"Photon-Gated Self-Organization".
(KORRIGIERTE VERSION: Reshape-Dimensionen angepasst)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Importiere die Simulations-Logik, um konsistente Daten zu erzeugen
try:
    from emergent_complex_sequences import CompetitiveSequenceGenerator, config as seq_config
    from run_experimentum_crucis import simulate_trial as run_crucis
except ImportError as e:
    print(f"Fehler: Konnte notwendige Skripte nicht importieren. Stelle sicher, dass sie im selben Verzeichnis liegen. Details: {e}")
    exit()


# --- Globales Setup ---
OUTPUT_DIR = 'final_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Erzeuge finale Plots für das Manuskript ---")
print(f"Speichere alle Plots in: '{OUTPUT_DIR}/'")

# =============================================================================
# Plot 1: Superradiance Power Scaling
# =============================================================================
def plot_superradiance_scaling():
    print("Erzeuge Plot 1: Superradiance Power Scaling...")
    M = np.logspace(1, 4, 100)
    P_incoherent = M
    P_coherent = M**2

    plt.figure(figsize=(10, 7))
    plt.loglog(M, P_incoherent, 'r--', label=r'Incoherent Sum (Noise): $P \propto M$')
    plt.loglog(M, P_coherent, 'b-', label=r'Coherent Sum (Superradiance): $P \propto M^2$')

    plt.title("Fig 1: The Power of Coherence - Bridging the Micro-Macro Gap")
    plt.xlabel("Number of Synchronized Emitters (M)")
    plt.ylabel("Total Emitted Power (Arbitrary Units, Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_1_superradiance_scaling.png'))
    plt.close()

# =============================================================================
# Plot 2: Experimentum Crucis
# =============================================================================
def plot_experimentum_crucis():
    print("Erzeuge Plot 2: Experimentum Crucis Simulation...")

    res_absent = run_crucis(blocker_present=False, seed=42)
    res_present = run_crucis(blocker_present=True, seed=42)

    # --- Plot für "Blocker Absent" (separat speichern) ---
    fig_absent, ax_absent = plt.subplots(figsize=(8, 5))
    lfp_plot_absent = res_absent["lfp"][-8000:]
    t_plot_absent = res_absent["t"][-8000:]
    ax_absent.plot(t_plot_absent, lfp_plot_absent, 'g-')
    ax_absent.axvline(res_absent["reset_time_ms"], color='r', linestyle='--', lw=2,
                      label=f'Fast Reset at t={res_absent["reset_time_ms"]:.2f} ms')
    ax_absent.set_title("A. Blocker Absent: Photon-Mediated Reset")
    ax_absent.set_xlabel("Time (ms)")
    ax_absent.set_ylabel("LFP Amplitude (a.u.)")
    ax_absent.legend()
    ax_absent.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_2a_experiment_blocker_absent.png'))
    plt.close(fig_absent)

    # --- Plot für "Blocker Present" (separat speichern) ---
    fig_present, ax_present = plt.subplots(figsize=(8, 5))
    lfp_plot_present = res_present["lfp"][-8000:]
    t_plot_present = res_present["t"][-8000:]
    ax_present.plot(t_plot_present, lfp_plot_present, 'g-')
    ax_present.axvline(res_present["reset_time_ms"], color='k', linestyle='--', lw=2,
                       label=f'Slow Reset at t={res_present["reset_time_ms"]:.2f} ms')
    ax_present.set_title("B. Blocker Present: Classical Effect Only")
    ax_present.set_xlabel("Time (ms)")
    ax_present.set_ylabel("LFP Amplitude (a.u.)")
    ax_present.legend()
    ax_present.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_2b_experiment_blocker_present.png'))
    plt.close(fig_present)


# =============================================================================
# Plot 3: Snapshot der Animation
# =============================================================================
def plot_animation_snapshot():
    print("Erzeuge Plot 3: Snapshot der Emergenz-Animation...")

    # KORREKTUR: Grid-Dimension dynamisch aus der Konfiguration ableiten
    num_digits = seq_config["network"]["num_digits"]
    grid_dim = int(np.sqrt(num_digits))
    if grid_dim**2 != num_digits:
        print(f"Warnung: num_digits ({num_digits}) ist keine perfekte Quadratzahl. Visualisierung könnte ungenau sein.")
        # Fallback auf die nächstliegende kleinere Quadratzahl
        grid_dim = int(np.floor(np.sqrt(num_digits)))


    # Hypothesis
    hyp_gen = CompetitiveSequenceGenerator(seq_config, seed=42)
    seq_hyp = hyp_gen.run_simulation("Hypothesis")
    W_hyp = hyp_gen.W
    activity_hyp = np.zeros(num_digits)
    activity_hyp[seq_hyp[seq_config["simulation"]["exploration_phase_steps"]+100]] = 1.0

    # Antithesis
    ant_gen = CompetitiveSequenceGenerator(seq_config, seed=42)
    seq_ant = ant_gen.run_simulation("Antithesis")
    W_ant = ant_gen.W
    activity_ant = np.zeros(num_digits)
    activity_ant[seq_ant[-100]] = 1.0

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Hypothesis Plots
    axs[0, 0].set_title("Hypothesis: Neuronal Activity")
    axs[0, 0].imshow(activity_hyp.reshape(grid_dim, grid_dim), cmap='hot', vmin=0, vmax=1)
    axs[1, 0].set_title("Hypothesis: Synaptic Weights")
    axs[1, 0].imshow(W_hyp, cmap='viridis', vmin=0, vmax=1.5)

    # Antithesis Plots
    axs[0, 1].set_title("Antithesis: Neuronal Activity")
    axs[0, 1].imshow(activity_ant.reshape(grid_dim, grid_dim), cmap='hot', vmin=0, vmax=1)
    axs[1, 1].set_title("Antithesis: Synaptic Weights")
    axs[1, 1].imshow(W_ant, cmap='viridis', vmin=0, vmax=1.5)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Fig 3: Emergent Dynamics - Coherent vs. Incoherent Gating", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_3_animation_snapshot.png'))
    plt.close()

# =============================================================================
# Plot 4: Quantitative Metriken
# =============================================================================
def plot_final_metrics():
    print("Erzeuge Plot 4: Quantitative Analyse der Komplexität...")

    all_results = []
    num_runs = 10
    for run_idx in range(num_runs):
        for condition in ["Hypothesis", "Antithesis"]:
            seed = 42 + run_idx * 2 + (0 if condition == "Hypothesis" else 1)
            generator = CompetitiveSequenceGenerator(seq_config, seed=seed)
            sequence = generator.run_simulation(condition)
            metrics = generator.analyze_sequence(sequence, seq_config["simulation"]["exploration_phase_steps"])
            all_results.append({"condition": condition, **metrics})

    df = pd.DataFrame(all_results)
    summary = df.groupby("condition").agg(
        lzc_mean=('final_lz_complexity', 'mean'),
        lzc_sem=('final_lz_complexity', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    summary.reindex(["Hypothesis", "Antithesis"]).plot(
        kind='bar', y='lzc_mean', yerr='lzc_sem', ax=ax, capsize=5,
        legend=False, color=['#1f77b4', '#ff7f0e']
    )
    ax.set_title("Fig 4: Emergent Complexity in the Exploitation Phase")
    ax.set_ylabel("Final Lempel-Ziv Complexity (Mean ± SEM)")
    ax.set_xlabel("Condition")
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_4_final_metrics.png'))
    plt.close()

# =============================================================================
# Main-Funktion
# =============================================================================
if __name__ == "__main__":
    plot_superradiance_scaling()
    plot_experimentum_crucis()
    plot_animation_snapshot()
    plot_final_metrics()
    print("\n--- Alle finalen Plots wurden erfolgreich erzeugt. ---")
