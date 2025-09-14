#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stufe 12: Visualisierung der Emergenz komplexer Sequenzen

Dieses Skript importiert die Logik aus 'emergent_complex_sequences.py'
und erzeugt eine Side-by-Side-Animation, die den dynamischen Unterschied
zwischen der Hypothese und der Antithese visualisiert.

LINKS: Hypothesis - Ein kohärenter Reward treibt das System zu komplexen,
       stabilen Pfaden.
RECHTS: Antithesis - Zufälliges Rauschen führt zu simplen, repetitiven
        Schleifen.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# Importiere die Kern-Logik aus dem vorherigen Skript
try:
    from emergent_complex_sequences import CompetitiveSequenceGenerator, config
except ImportError:
    print("Fehler: Stelle sicher, dass 'emergent_complex_sequences.py' im selben Verzeichnis liegt.")
    exit()

# =============================================================================
# Simulations-Wrapper
# =============================================================================

def run_full_simulation(condition: str, cfg: dict, seed: int) -> tuple[np.ndarray, list]:
    """Führt eine Simulation durch und speichert die komplette Historie."""

    generator = CompetitiveSequenceGenerator(cfg, seed=seed)

    # Historie für die Animation speichern
    weight_history = []
    activity_history = []

    current_digit = generator.rng.integers(0, generator.num_digits)
    current_path: list = []

    p_cfg = cfg["plasticity"]

    for step in range(cfg["simulation"]["num_steps"]):
        # Zustand vor dem Schritt speichern
        weight_history.append(generator.W.copy())

        activity = np.zeros(generator.num_digits)
        activity[current_digit] = 1.0
        activity_history.append(activity)

        # Logik aus dem Original-Skript übernehmen
        generator.E *= p_cfg["eligibility_decay"]
        generator.H *= p_cfg["habituation_decay"]

        in_exploitation_phase = step >= cfg["simulation"]["exploration_phase_steps"]
        next_digit = generator.generate_step(current_digit, in_exploitation_phase)

        transition = (current_digit, next_digit)
        generator.E[current_digit, next_digit] += 1.0
        generator.H[current_digit, next_digit] = p_cfg["habituation_strength"]
        current_path.append(transition)

        reward = 0.0
        if condition == "Hypothesis":
            if in_exploitation_phase:
                if len(current_path) > 0 and tuple(current_path) == tuple(generator.best_path_found[:len(current_path)]):
                    reward = (len(current_path) / len(generator.best_path_found))**2
            else:
                if tuple(current_path) not in generator.visited_paths_in_explore:
                    reward = len(current_path) / cfg["simulation"]["exploration_phase_steps"]
                    if len(current_path) > len(generator.best_path_found):
                        generator.best_path_found = list(current_path)
        else: # Antithesis
            if generator.rng.random() < 0.1:
                reward = generator.rng.random() * 0.2

        if reward > 0:
            lr = p_cfg["learning_rate_exploit"] if in_exploitation_phase else p_cfg["learning_rate_explore"]
            dw = lr * generator.E * reward
            generator.W += dw

            if not in_exploitation_phase:
                generator.visited_paths_in_explore.add(tuple(current_path))

            if reward >= 1.0 or not in_exploitation_phase:
                current_path = []

        current_digit = next_digit

    np.fill_diagonal(generator.W, 0)
    return np.array(weight_history), np.array(activity_history)

# =============================================================================
# Visualisierung
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Animation der emergenten Sequenz-Dynamik.")
    parser.add_argument("--seed", type=int, default=42, help="Seed für den Zufallsgenerator.")
    args = parser.parse_args()

    print("Führe Simulationen für die Animation durch (dies kann einen Moment dauern)...")

    # Führe beide Simulationen durch und sammle die Daten
    W_hyp, A_hyp = run_full_simulation("Hypothesis", config, seed=args.seed)
    W_ant, A_ant = run_full_simulation("Antithesis", config, seed=args.seed)

    print("Erzeuge Side-by-Side-Animation...")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1]})

    # --- Setup für Hypothesis (linke Spalte) ---
    axs[0, 0].set_title("Hypothesis: Neuronal Activity")
    activity_grid_hyp = A_hyp[0].reshape(16, 16)
    im_act_hyp = axs[0, 0].imshow(activity_grid_hyp, cmap='hot', vmin=0, vmax=1)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[1, 0].set_title("Hypothesis: Synaptic Weights (W)")
    im_w_hyp = axs[1, 0].imshow(W_hyp[0], cmap='viridis', vmin=0, vmax=1.2)
    axs[1, 0].set_xlabel("To Neuron")
    axs[1, 0].set_ylabel("From Neuron")

    # --- Setup für Antithesis (rechte Spalte) ---
    axs[0, 1].set_title("Antithesis: Neuronal Activity")
    activity_grid_ant = A_ant[0].reshape(16, 16)
    im_act_ant = axs[0, 1].imshow(activity_grid_ant, cmap='hot', vmin=0, vmax=1)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 1].set_title("Antithesis: Synaptic Weights (W)")
    im_w_ant = axs[1, 1].imshow(W_ant[0], cmap='viridis', vmin=0, vmax=1.2)
    axs[1, 1].set_xlabel("To Neuron")
    axs[1, 1].set_yticks([])

    # Zeit-Titel
    time_text = fig.suptitle(f"Time Step: 0 / {config['simulation']['num_steps']}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        # Update Aktivität
        im_act_hyp.set_data(A_hyp[frame].reshape(16, 16))
        im_act_ant.set_data(A_ant[frame].reshape(16, 16))

        # Update Gewichte
        im_w_hyp.set_data(W_hyp[frame])
        im_w_ant.set_data(W_ant[frame])

        phase = "Exploration" if frame < config["simulation"]["exploration_phase_steps"] else "Exploitation"
        time_text.set_text(f"Time Step: {frame} / {config['simulation']['num_steps']} | Phase: {phase}")

        return im_act_hyp, im_w_hyp, im_act_ant, im_w_ant, time_text

    # Animation erstellen
    ani = animation.FuncAnimation(fig, update, frames=config['simulation']['num_steps'], blit=True, interval=20)

    filename = "emergent_sequence_side_by_side_10k.mp4"
    ani.save(filename, writer='ffmpeg', dpi=120, progress_callback=lambda i, n: print(f'  -> Saving frame {i+1} of {n}'))

    print(f"\n[OK] Animation erfolgreich gespeichert: {filename}")

if __name__ == "__main__":
    main()
