#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stufe 5: Emergenz von symbolischen Sequenzen & quantitative Analyse

Dieses Skript simuliert ein Netzwerk, das eine Zahlenkette (z.B. 1->2->3->4)
lernt. Das Lernen wird durch ein "Binding"-Signal gesteuert, das durch
UPEs repräsentiert wird.

HYPOTHESE:
Ein kohärenter UPE-Burst, der durch die Koinzidenz-Aktivität zweier Neuronen
ausgelöst wird, ermöglicht die Stärkung der synaptischen Verbindung zwischen
ihnen (Hebb'sches Lernen).

ANTITHESE:
Inkohärente, verrauschte UPEs liefern kein präzises Timing-Signal.
Das Netzwerk kann die geordnete Sequenz nicht lernen.

Das Skript führt automatisch eine statistische Analyse über mehrere Durchläufe
durch und berichtet quantitative Metriken (Lernerfolg, Sequenz-Genauigkeit).
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# =============================================================================
# 1. Konfiguration
# =============================================================================
config = {
    "network": {
        "num_digits": 4, # Anzahl der "Zahlen"-Neuronen (z.B. 1, 2, 3, 4)
    },
    "simulation": {
        "training_epochs": 100,
        "sequence_to_learn": [0, 1, 2, 3], # Neuron-Indizes der Zielsequenz
        "noise_level": 0.1, # Grundaktivität
    },
    "plasticity": {
        "learning_rate": 0.1,
        "ltp_threshold_coherent": 0.9, # Hohe Schwelle, nur bei starkem Burst
        "ltp_threshold_incoherent": 0.1, # Niedrige Schwelle für verrauschte UPEs
    },
    "statistics": {
        "num_runs": 10,
        "base_seed": 42,
    }
}

# =============================================================================
# 2. Simulations-Kern
# =============================================================================

class SequenceNetwork:
    """Ein Netzwerk, das lernt, Sequenzen zu erzeugen."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.num_digits = cfg["network"]["num_digits"]

        # Übergangsmatrix: W[i, j] ist die Stärke der Verbindung von Neuron i zu j
        self.W = self.rng.uniform(0.0, 0.1, size=(self.num_digits, self.num_digits))
        np.fill_diagonal(self.W, 0) # Keine Selbst-Verbindungen

    def train(self, condition: str):
        """Trainiert das Netzwerk für eine gegebene Anzahl von Epochen."""

        target_sequence = self.cfg["simulation"]["sequence_to_learn"]
        p_cfg = self.cfg["plasticity"]

        for _ in range(self.cfg["simulation"]["training_epochs"]):
            # In jeder Epoche wird ein Paar aus der Sequenz "aktiviert"
            idx = self.rng.integers(0, len(target_sequence) - 1)
            pre_neuron = target_sequence[idx]
            post_neuron = target_sequence[idx + 1]

            # Simuliere Koinzidenz-Aktivität
            pre_activity = 1.0
            post_activity = 1.0

            # Simuliere das UPE-"Binding"-Signal
            if condition == "Hypothesis":
                # Kohärenter Burst: starkes, zuverlässiges Signal
                upe_signal = self.rng.uniform(0.9, 1.1)
                threshold = p_cfg["ltp_threshold_coherent"]
            else: # Antithesis
                # Inkohärentes Rauschen: schwaches, unzuverlässiges Signal
                upe_signal = self.rng.uniform(0.0, 0.2)
                threshold = p_cfg["ltp_threshold_incoherent"]

            # Hebb'sche Lernregel, "ge-gated" durch das UPE-Signal
            if pre_activity > 0 and post_activity > 0 and upe_signal > threshold:
                # Long-Term Potentiation (LTP)
                dw = p_cfg["learning_rate"] * (1.0 - self.W[pre_neuron, post_neuron])
                self.W[pre_neuron, post_neuron] += dw

        self.W = np.clip(self.W, 0, 1.0)

    def recall_sequence(self, start_digit: int, max_len: int = 10) -> List[int]:
        """Erzeugt eine Sequenz aus dem gelernten Wissen."""

        sequence = [start_digit]
        current_digit = start_digit

        for _ in range(max_len - 1):
            # Nächstes Neuron wird probabilistisch basierend auf Gewichten gewählt
            next_probs = self.W[current_digit, :]

            # Füge Grundrauschen hinzu
            next_probs = next_probs + self.rng.uniform(0, self.cfg["simulation"]["noise_level"], size=self.num_digits)

            # Verhindere, dass das Neuron sofort zu sich selbst zurückkehrt
            next_probs[current_digit] = -np.inf

            if np.sum(next_probs) == -np.inf:
                break

            next_digit = np.argmax(next_probs)
            sequence.append(next_digit)
            current_digit = next_digit

        return sequence

    def evaluate(self) -> float:
        """Quantifiziert die Genauigkeit der gelernten Sequenz."""
        target_sequence = self.cfg["simulation"]["sequence_to_learn"]
        recalled_sequence = self.recall_sequence(start_digit=target_sequence[0])

        # Levenshtein-Distanz als Maß für die Sequenz-Ähnlichkeit
        n, m = len(recalled_sequence), len(target_sequence)
        dp = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if recalled_sequence[i - 1] == target_sequence[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1,        # Deletion
                               dp[i, j - 1] + 1,        # Insertion
                               dp[i - 1, j - 1] + cost) # Substitution

        max_len = max(n, m)
        accuracy = (max_len - dp[n, m]) / max_len if max_len > 0 else 1.0
        return accuracy

# =============================================================================
# 3. Main-Funktion mit statistischer Analyse
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stufe 5: Emergenz symbolischer Sequenzen.")
    parser.add_argument("--plots", action="store_true", help="Generiere PNG-Plots der finalen Ergebnisse.")
    args = parser.parse_args()

    conditions = ["Hypothesis", "Antithesis"]
    all_results = []

    print("Führe statistische Analyse des Sequenz-Lernens durch...")
    with tqdm(total=config["statistics"]["num_runs"] * len(conditions)) as pbar:
        for run_idx in range(config["statistics"]["num_runs"]):
            pbar.set_description(f"Run {run_idx + 1}/{config['statistics']['num_runs']}")
            for condition_name in conditions:
                exp_seed = config["statistics"]["base_seed"] + run_idx * len(conditions) + conditions.index(condition_name)

                network = SequenceNetwork(config, seed=exp_seed)
                network.train(condition=condition_name)
                accuracy = network.evaluate()

                all_results.append({
                    "run": run_idx + 1,
                    "condition": condition_name,
                    "accuracy": accuracy,
                })
                pbar.update(1)

    # --- Finale Auswertung ---
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby("condition").agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_sem=('accuracy', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
    )

    print("\n" + "="*80)
    print(f"        FINALE AUSWERTUNG (MITTELWERTE ± SEM ÜBER {config['statistics']['num_runs']} DURCHLÄUFE)")
    print("="*80)
    # Sicherstellen, dass die Reihenfolge konsistent ist
    print(summary.reindex(conditions).to_string(float_format="%.3f"))

    # --- Schlussfolgerung ---
    hyp_acc = summary.loc["Hypothesis", "accuracy_mean"]
    ant_acc = summary.loc["Antithesis", "accuracy_mean"]

    print("\n[SCHLUSSFOLGERUNG]")
    if hyp_acc > 0.8 and ant_acc < 0.5 and (hyp_acc - ant_acc) > 0.3:
        print("Die Simulation stützt die Hypothese stark: Nur der kohärente UPE-Burst")
        print("ermöglichte als präzises 'Binding'-Signal das zuverlässige Lernen der")
        print("symbolischen Sequenz. Die emergente Zahlenkette ist kausal von der")
        print("Kohärenz des UPE-Signals abhängig.")
        print("Die Antithese, dass UPEs nur irrelevantes Rauschen sind, wurde falsifiziert.")
    else:
        print("Der Unterschied zwischen den Bedingungen war nicht signifikant genug,")
        print("um die Hypothese eindeutig zu stützen.")

    if args.plots:
        # Erzeuge einen Beispiel-Plot der gelernten Übergangsmatrix
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Gelernte Übergangsmatrizen (Wärmebild)", fontsize=16)

        # Hypothesis
        net_hyp = SequenceNetwork(config, seed=config["statistics"]["base_seed"])
        net_hyp.train("Hypothesis")
        im1 = axs[0].imshow(net_hyp.W, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title("Hypothesis (kohärenter Burst)")
        axs[0].set_xlabel("Zu Neuron")
        axs[0].set_ylabel("Von Neuron")
        axs[0].set_xticks(np.arange(config["network"]["num_digits"]))
        axs[0].set_yticks(np.arange(config["network"]["num_digits"]))

        # Antithesis
        net_ant = SequenceNetwork(config, seed=config["statistics"]["base_seed"])
        net_ant.train("Antithesis")
        im2 = axs[1].imshow(net_ant.W, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title("Antithesis (inkohärentes Rauschen)")
        axs[1].set_xlabel("Zu Neuron")
        axs[1].set_yticks([]) # Keine y-Achsenbeschriftung

        fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.8, label="Synaptische Stärke")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("emergent_sequence_weights.png")
        print("\n[OK] Plot der Gewichtsmatrizen gespeichert.")

if __name__ == "__main__":
    main()
