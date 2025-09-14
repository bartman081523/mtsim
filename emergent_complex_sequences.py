#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stufe 11: Finale Validierung mit korrigierter Komplexitäts-Metrik
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm

# =============================================================================
# 1. Konfiguration
# =============================================================================
config = {
    "network": {
        "num_digits": 256,
    },
    "simulation": {
        "num_steps": 10000,
        "exploration_phase_steps": 5000,
    },
    "plasticity": {
        "learning_rate_explore": 0.2,
        "learning_rate_exploit": 0.5,
        "eligibility_decay": 0.95,
        "habituation_decay": 0.8,
        "habituation_strength": -1.0,
        "attractor_strength": 0.3,
    },
    "statistics": {
        "num_runs": 10,
        "base_seed": 42,
    }
}

# =============================================================================
# 2. Simulations-Kern
# =============================================================================

class CompetitiveSequenceGenerator:
    """Ein Netzwerk, das lernt, komplexe Sequenzen zu finden und zu stabilisieren."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.num_digits = cfg["network"]["num_digits"]

        self.W = self.rng.uniform(0.1, 0.2, size=(self.num_digits, self.num_digits))
        np.fill_diagonal(self.W, -np.inf)

        self.H = np.zeros_like(self.W)
        self.E = np.zeros_like(self.W)

        self.best_path_found: List[Tuple[int, int]] = []
        self.visited_paths_in_explore = set()

    def generate_step(self, current_digit: int, in_exploitation: bool) -> int:
        effective_weights = self.W.copy() + self.H

        if in_exploitation and self.best_path_found:
            for pre, post in self.best_path_found:
                effective_weights[pre, post] += self.cfg["plasticity"]["attractor_strength"]

        probs = effective_weights[current_digit, :]
        temp = 0.02 if in_exploitation else 0.1

        probs_exp = np.exp((probs - np.max(probs)) / temp)
        probs_exp[np.isneginf(probs)] = 0
        sum_exp = np.sum(probs_exp)

        if sum_exp > 1e-9:
            final_probs = probs_exp / sum_exp
            next_digit = self.rng.choice(self.num_digits, p=final_probs)
        else:
            valid_choices = np.where(np.isfinite(probs))[0]
            if not len(valid_choices): return (current_digit + 1) % self.num_digits
            next_digit = self.rng.choice(valid_choices)
        return next_digit

    def run_simulation(self, condition: str):
        current_digit = self.rng.integers(0, self.num_digits)
        sequence = [current_digit]
        current_path: List[Tuple[int, int]] = []

        p_cfg = self.cfg["plasticity"]

        for step in range(self.cfg["simulation"]["num_steps"] - 1):
            self.E *= p_cfg["eligibility_decay"]
            self.H *= p_cfg["habituation_decay"]

            in_exploitation_phase = step >= self.cfg["simulation"]["exploration_phase_steps"]
            next_digit = self.generate_step(current_digit, in_exploitation_phase)
            sequence.append(next_digit)

            transition = (current_digit, next_digit)
            self.E[current_digit, next_digit] += 1.0
            self.H[current_digit, next_digit] = p_cfg["habituation_strength"]
            current_path.append(transition)

            reward = 0.0

            if condition == "Hypothesis":
                if in_exploitation_phase:
                    if len(current_path) > 0 and tuple(current_path) == tuple(self.best_path_found[:len(current_path)]):
                         reward = (len(current_path) / len(self.best_path_found))**2
                else:
                    if tuple(current_path) not in self.visited_paths_in_explore:
                        reward = len(current_path) / self.cfg["simulation"]["exploration_phase_steps"]
                        if len(current_path) > len(self.best_path_found):
                            self.best_path_found = list(current_path)
            else: # Antithesis
                reward = len(current_path) / self.cfg["simulation"]["num_steps"] * self.rng.random()

            if reward > 0:
                lr = p_cfg["learning_rate_exploit"] if in_exploitation_phase else p_cfg["learning_rate_explore"]
                dw = lr * self.E * reward
                self.W += dw

                if not in_exploitation_phase:
                    self.visited_paths_in_explore.add(tuple(current_path))

                if reward >= 1.0 or not in_exploitation_phase:
                    current_path = []

            current_digit = next_digit

        self.W = np.clip(self.W, 0, 2.0)
        np.fill_diagonal(self.W, -np.inf)
        return sequence

    @staticmethod
    def analyze_sequence(sequence: List[int], exploitation_start_step: int) -> Dict[str, float]:
        """Quantifiziert die Komplexität und Ordnung der finalen Sequenz."""

        final_sequence = sequence[exploitation_start_step:]
        n = len(final_sequence)

        if n < 2:
            return {"final_lz_complexity": 0, "unique_bigrams_final": 0}

        # 1. Lempel-Ziv-Komplexität der finalen Sequenz
        sub_sequences = set()
        i = 0
        while i < n:
            j = i + 1
            while j <= n and tuple(final_sequence[i:j]) in sub_sequences:
                j += 1
            sub_sequences.add(tuple(final_sequence[i:j]))
            i = j
        lz_complexity = len(sub_sequences)

        # 2. Anzahl einzigartiger Übergänge in der finalen Sequenz
        unique_bigrams = len(set(zip(final_sequence, final_sequence[1:])))

        return {
            "final_lz_complexity": lz_complexity,
            "unique_bigrams_final": unique_bigrams,
        }

# =============================================================================
# 3. Main-Funktion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stufe 11: Finale Validierung mit korrigierter Metrik.")
    parser.add_argument("--plots", action="store_true", help="Generiere PNG-Plots der finalen Ergebnisse.")
    args = parser.parse_args()

    conditions = ["Hypothesis", "Antithesis"]
    all_results = []

    print("Führe statistische Analyse der Sequenz-Dynamik durch...")
    with tqdm(total=config["statistics"]["num_runs"] * len(conditions)) as pbar:
        for run_idx in range(config["statistics"]["num_runs"]):
            pbar.set_description(f"Run {run_idx + 1}/{config['statistics']['num_runs']}")
            for condition_name in conditions:
                exp_seed = config["statistics"]["base_seed"] + run_idx * len(conditions) + conditions.index(condition_name)

                generator = CompetitiveSequenceGenerator(config, seed=exp_seed)
                sequence = generator.run_simulation(condition=condition_name)
                metrics = generator.analyze_sequence(sequence, config["simulation"]["exploration_phase_steps"])

                if run_idx == 0:
                    print(f"\nBeispiel-Sequenz ({condition_name}, Run 1):")
                    print("... " + " -> ".join(map(str, sequence)))

                all_results.append({
                    "run": run_idx + 1,
                    "condition": condition_name,
                    "final_lz_complexity": metrics["final_lz_complexity"],
                    "unique_bigrams_final": metrics["unique_bigrams_final"],
                })
                pbar.update(1)

    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby("condition").agg(
        lz_complexity_mean=('final_lz_complexity', 'mean'),
        lz_complexity_sem=('final_lz_complexity', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
        unique_bigrams_mean=('unique_bigrams_final', 'mean'),
    )

    print("\n" + "="*80)
    print(f"        FINALE AUSWERTUNG (MITTELWERTE ± SEM ÜBER {config['statistics']['num_runs']} DURCHLÄUFE)")
    print("="*80)
    print(summary.reindex(conditions).to_string(float_format="%.2f"))

    hyp_lz = summary.loc["Hypothesis", "lz_complexity_mean"]
    ant_lz = summary.loc["Antithesis", "lz_complexity_mean"]

    print("\n[SCHLUSSFOLGERUNG]")
    if hyp_lz > (ant_lz * 1.2): # Wenn Hypothese >20% komplexer ist
        print("Die Simulation stützt die Hypothese stark: Das System, das durch kohärente")
        print("UPEs belohnt wird, erzeugt in der Exploitationsphase signifikant komplexere")
        print("und abwechslungsreichere Sequenzen. Es hat gelernt, einen reichhaltigen,")
        print("stabilen Pfad zu explorieren.")
        print("Die Antithese verfällt in simple, repetitive Schleifen mit geringer Komplexität.")
    else:
        print("Der Unterschied in der finalen Komplexität war nicht signifikant genug,")
        print("um die Hypothese eindeutig zu stützen.")

    if args.plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        summary.reindex(conditions).plot(
            kind='bar', y='lz_complexity_mean', yerr='lz_complexity_sem', ax=ax,
            capsize=4, legend=False, title="Emergente Komplexität in der Exploitationsphase (Höher ist besser)"
        )
        ax.set_ylabel("Finale Lempel-Ziv-Komplexität")
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        plt.savefig("emergent_complexity_results_v6.png")
        print("\n[OK] Plot der Komplexität gespeichert.")

if __name__ == "__main__":
    main()
