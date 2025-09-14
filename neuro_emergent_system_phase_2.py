#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
neuro_emergent_system_phase_2.py

Stufe 3/4: Emergenz & Mechanistische Validierung

Dieses Skript simuliert und vergleicht drei Hypothesen zur Ermöglichung von
assoziativem Lernen (Engram-Bildung) in einem neuronalen Netzwerk:

1. HYPOTHESE (Superradiance): Ein ultrakurzer, globaler Photonen-Burst fungiert
   als perfektes "Learning Gate", das Hebb'sche Plastizität präzise steuert.

2. BASELINE 1 (Gamma-Oszillation): Ein etablierter neurobiologischer Mechanismus,
   bei dem eine 40-Hz-Netzwerk-Oszillation Lernfenster bereitstellt.

3. BASELINE 2 (Klassischer Puls): Ein idealisierter, scharfer elektrischer Puls
   von einem Interneuron, der ebenfalls als globales Signal dient.

ZIEL:
Zu zeigen, ob die Superradiance-Hypothese einen quantifizierbaren Vorteil
(z.B. schnellere Lernzeit, höhere Engramm-Qualität) gegenüber den starken,
etablierten Alternativen bietet.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import pandas as pd

# =============================================================================
# 1. Konfiguration des Experiments
# =============================================================================
config = {
    "simulation": {
        "t_sim_ms": 100.0,
        "dt_ms": 0.1,
        "seed": 42,
    },
    "network": {
        "n_input": 50,
        "n_engram": 20,
    },
    "lif_neuron": {
        "V_rest": -70.0, "V_th": -55.0, "V_reset": -75.0,
        "tau_m": 10.0, "R_m": 10.0,
    },
    "plasticity": {
        "eta_ltp": 0.01,  # Lernrate
        "tau_stdp": 15.0,  # STDP-Zeitfenster in ms
    },
    "stimulus": {
        "rate_hz": 50,
        "duration_ms": 60,
        "start_ms": 20,
        "current_nA": 2.5, # KORRIGIERT: Erhöht, damit Neuronen feuern
    },
    "learning_signal": {
        "sr_burst_duration_ms": 0.1,
        "classical_pulse_duration_ms": 1.0,
        "gamma_freq_hz": 40.0,
        "signal_current_nA": 4.0, # KORRIGIERT: Erhöht für stärkere Wirkung
    },
    "training": {
        "max_epochs": 50,
        "target_selectivity": 0.95,
    }
}

# =============================================================================
# 2. Modulare System-Komponenten
# =============================================================================

class StimulusGenerator:
    """Erzeugt verschiedene Eingangsmuster."""
    def __init__(self, cfg: Dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.time = np.arange(0, cfg["simulation"]["t_sim_ms"], cfg["simulation"]["dt_ms"])

        # Definiere Muster-Indizes
        n_input = cfg["network"]["n_input"]
        self.pattern_A = np.arange(0, n_input // 2)
        self.pattern_B = np.arange(n_input // 2, n_input)
        self.pattern_A_partial = self.rng.choice(
            self.pattern_A, size=len(self.pattern_A) // 2, replace=False
        )
        self.pattern_random = self.rng.choice(
            n_input, size=len(self.pattern_A), replace=False
        )

    def get_pattern(self, name: str) -> np.ndarray:
        """Gibt den Strom für ein benanntes Muster zurück."""
        s_cfg = self.cfg["stimulus"]
        n_cfg = self.cfg["network"]

        indices = {
            "A": self.pattern_A,
            "B": self.pattern_B,
            "A_partial": self.pattern_A_partial,
            "random": self.pattern_random
        }[name]

        current = np.zeros((n_cfg["n_input"], len(self.time)))
        prob = s_cfg["rate_hz"] * (self.cfg["simulation"]["dt_ms"] / 1000.0)
        mask = (self.time >= s_cfg["start_ms"]) & (self.time < s_cfg["start_ms"] + s_cfg["duration_ms"])

        for i in indices:
            spikes = self.rng.random(np.sum(mask)) < prob
            current[i, mask] = spikes * s_cfg["current_nA"]
        return current


class LIFNetwork:
    """Simuliert das LIF-Netzwerk und wendet eine Plastizitätsregel an."""
    def __init__(self, cfg: Dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_cfg = cfg["network"]
        self.l_cfg = cfg["lif_neuron"]

        self.W = self.rng.uniform(0.1, 0.4, size=(self.n_cfg["n_engram"], self.n_cfg["n_input"]))
        self.W_initial = self.W.copy()

    def reset_weights(self):
        self.W = self.W_initial.copy()

    def run(self, input_current: np.ndarray, global_signal: np.ndarray = None, apply_plasticity: bool = False):
        """Führt eine Simulation durch."""
        time = np.arange(0, self.cfg["simulation"]["t_sim_ms"], self.cfg["simulation"]["dt_ms"])
        dt_ms = self.cfg["simulation"]["dt_ms"]

        V = np.full(self.n_cfg["n_engram"], self.l_cfg["V_rest"])
        last_spike_input = np.full(self.n_cfg["n_input"], -np.inf)
        last_spike_engram = np.full(self.n_cfg["n_engram"], -np.inf)
        spikes_engram = []

        for i, t in enumerate(time):
            I_syn = self.W @ input_current[:, i]
            I_global = global_signal[i] if global_signal is not None else 0

            dV = (-(V - self.l_cfg["V_rest"]) + self.l_cfg["R_m"] * (I_syn + I_global)) / self.l_cfg["tau_m"]
            V += dV * dt_ms

            spiked_neurons = np.where(V >= self.l_cfg["V_th"])[0]

            if len(spiked_neurons) > 0:
                V[spiked_neurons] = self.l_cfg["V_reset"]
                for n_idx in spiked_neurons:
                    spikes_engram.append((t, n_idx))
                    last_spike_engram[n_idx] = t

                    if apply_plasticity:
                        self._apply_stdp(t, n_idx, last_spike_input)

            active_inputs = np.where(input_current[:, i] > 0)[0]
            last_spike_input[active_inputs] = t

        return spikes_engram

    def _apply_stdp(self, t_post: float, n_idx: int, last_spike_input: np.ndarray):
        """Wendet eine einfache STDP-Regel an."""
        p_cfg = self.cfg["plasticity"]
        time_diffs = t_post - last_spike_input

        # LTP für Inputs, die kurz vor dem Engram-Neuron gefeuert haben
        causal_inputs = np.where((time_diffs > 0) & (time_diffs < p_cfg["tau_stdp"]))[0]

        for in_idx in causal_inputs:
            dw = p_cfg["eta_ltp"] * np.exp(-time_diffs[in_idx] / p_cfg["tau_stdp"]) * (1.0 - self.W[n_idx, in_idx])
            self.W[n_idx, in_idx] += dw

        self.W = np.clip(self.W, 0, 1.2) # Gewichte begrenzen


# =============================================================================
# 3. Experiment-Klasse
# =============================================================================

class Experiment:
    """Führt das Training und die Evaluation für eine Bedingung durch."""
    def __init__(self, name: str, cfg: Dict, stimulus_gen: StimulusGenerator):
        self.name = name
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["simulation"]["seed"])
        self.stimulus_gen = stimulus_gen
        self.network = LIFNetwork(cfg, self.rng)

        self.time = np.arange(0, cfg["simulation"]["t_sim_ms"], cfg["simulation"]["dt_ms"])
        self.global_signal = self._create_global_signal()

    def _create_global_signal(self) -> np.ndarray:
        """Erzeugt das spezifische Lernsignal für diese Bedingung."""
        s_cfg = self.cfg["learning_signal"]
        t_center = self.cfg["simulation"]["t_sim_ms"] / 2
        signal = np.zeros_like(self.time)

        if self.name == "Superradiance":
            duration = s_cfg["sr_burst_duration_ms"]
            mask = (self.time > t_center) & (self.time < t_center + duration)
            signal[mask] = s_cfg["signal_current_nA"]
        elif self.name == "Classical Pulse":
            duration = s_cfg["classical_pulse_duration_ms"]
            mask = (self.time > t_center) & (self.time < t_center + duration)
            signal[mask] = s_cfg["signal_current_nA"]
        elif self.name == "Gamma Oscillation":
            freq = s_cfg["gamma_freq_hz"]
            signal = s_cfg["signal_current_nA"] * (1 + np.sin(2 * np.pi * freq / 1000 * self.time)) / 2

        return signal

    def train(self) -> Dict:
        """Trainiert das Netzwerk, bis das Ziel erreicht ist."""
        print(f"\n--- Training Condition: {self.name} ---")
        input_A = self.stimulus_gen.get_pattern("A")

        for epoch in range(1, self.cfg["training"]["max_epochs"] + 1):
            _ = self.network.run(input_A, self.global_signal, apply_plasticity=True)

            selectivity, _, _ = self.evaluate()
            print(f"Epoch {epoch:02d}: Selectivity = {selectivity:.3f}")

            if selectivity >= self.cfg["training"]["target_selectivity"]:
                print(f"Target selectivity reached in {epoch} epochs.")
                return {"epochs_to_criterion": epoch}

        print("Max epochs reached without meeting criterion.")
        return {"epochs_to_criterion": self.cfg["training"]["max_epochs"]}

    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluiert die Leistung des trainierten Netzwerks."""
        resp_A = len(self.network.run(self.stimulus_gen.get_pattern("A")))
        resp_B = len(self.network.run(self.stimulus_gen.get_pattern("B")))
        resp_partial = len(self.network.run(self.stimulus_gen.get_pattern("A_partial")))
        resp_random = len(self.network.run(self.stimulus_gen.get_pattern("random")))

        selectivity = (resp_A - resp_B) / (resp_A + resp_B + 1e-9)
        completion = resp_partial
        engram_snr = resp_A / (resp_random + 1e-9)

        return selectivity, completion, engram_snr

# =============================================================================
# 4. Main-Funktion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Vergleichende Simulation von Lernmechanismen.")
    parser.add_argument("--plots", action="store_true", help="Generiere PNG-Plots der Ergebnisse.")
    args = parser.parse_args()

    # Seed für das gesamte Experiment setzen
    np.random.seed(config["simulation"]["seed"])
    rng = np.random.default_rng(config["simulation"]["seed"])
    stimulus_gen = StimulusGenerator(config, rng)

    conditions = ["Superradiance", "Gamma Oscillation", "Classical Pulse"]
    results = {}

    for condition_name in conditions:
        # Für jede Bedingung einen eigenen Seed nutzen, um Fairness zu gewährleisten
        exp_seed = config["simulation"]["seed"] + conditions.index(condition_name)
        config["simulation"]["seed"] = exp_seed

        exp = Experiment(condition_name, config, stimulus_gen)

        training_results = exp.train()
        post_selectivity, post_completion, post_snr = exp.evaluate()

        results[condition_name] = {
            "epochs_to_criterion": training_results["epochs_to_criterion"],
            "post_selectivity": post_selectivity,
            "post_completion": post_completion,
            "post_engram_snr": post_snr,
            "final_weights": exp.network.W.copy()
        }

    # --- Finale Auswertung und Schlussfolgerung ---
    print("\n" + "="*60)
    print("        FINALE AUSWERTUNG & VERGLEICH")
    print("="*60)

    summary_df = pd.DataFrame(results).T
    print(summary_df[['epochs_to_criterion', 'post_selectivity', 'post_completion', 'post_engram_snr']].to_string(float_format="%.2f"))

    best_condition = summary_df['epochs_to_criterion'].idxmin()

    print("\n[SCHLUSSFOLGERUNG]")
    print(f"Die effizienteste Lernbedingung war '{best_condition}' "
          f"(benötigte {summary_df.loc[best_condition, 'epochs_to_criterion']} Epochen).")

    if best_condition == "Superradiance":
        print("Die Simulation stützt die Hypothese, dass ein ultrakurzer, präziser Burst einen")
        print("signifikanten Vorteil bei der schnellen und spezifischen Bildung von neuronalen")
        print("Engrammen bietet, selbst im Vergleich zu starken neurobiologischen Alternativen.")
    else:
        print("Die Superradiance-Hypothese zeigte keinen klaren Vorteil gegenüber der etablierten")
        print(f"Alternative '{best_condition}'. Dies schwächt die Behauptung, dass ein 'Quanten'-Signal")
        print("notwendig oder überlegen ist, um assoziatives Lernen zu ermöglichen.")

    if args.plots:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Finale Gewichtsmatrizen nach dem Training", fontsize=16)
        for i, name in enumerate(conditions):
            im = axs[i].imshow(results[name]["final_weights"], aspect='auto', cmap='magma', vmin=0, vmax=1)
            axs[i].set_title(name)
            axs[i].set_xlabel("Input Neurons")
        axs[0].set_ylabel("Engram Neurons")
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("neuro_emergent_system_phase_2_weights.png")
        print("\n[OK] Plot der Gewichte gespeichert.")

if __name__ == "__main__":
    main()
