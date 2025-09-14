#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stufe 3: Emergenz neurobiologischer Eigenschaften

Dieses Skript simuliert ein neuronales Netzwerk, das eine elementare Form von
assoziativem Gedächtnis ausbildet. Eine Gruppe von Neuronen ("Engram") lernt,
selektiv auf ein bestimmtes Eingangsmuster ("Pattern A") zu feuern.

HYPOTHESE:
Das Lernen (die Verstärkung der richtigen Synapsen via STDP) wird erst durch einen
zeitlich präzisen, superradianten Photonen-Burst ermöglicht. Der Burst wirkt als
globales Synchronisationssignal ("common input"), das die für die Plastizität
notwendige Koinzidenz von prä- und postsynaptischem Feuern erzwingt.

ANTITHESE:
Ein inkohärenter Photonen-Puls mit der gleichen Gesamtenergie, aber ohne zeitliche
Präzision, reicht nicht aus, um ein stabiles Engram zu formen. Die Photonen sind
nur irrelevantes Rauschen.

EXPERIMENT:
1. Pre-Training: Teste die Antwort des naiven Netzes auf Pattern A und B.
2. Training:
   - HYP-Netz: Präsentiere Pattern A + kohärenten SR-Burst.
   - ANT-Netz: Präsentiere Pattern A + inkohärenten Puls.
3. Post-Training: Teste beide Netze erneut auf A, B und ein unvollständiges A.

ERWARTUNG:
Nur das HYP-Netz wird eine hohe Selektivität für Pattern A entwickeln und fähig
sein, das unvollständige Muster zu vervollständigen.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

# =============================================================================
# 1. System-Komponenten
# =============================================================================

class PhotonicSystem:
    """Erzeugt entweder einen kohärenten Burst oder inkohärentes Rauschen."""

    def __init__(self, t_sim_ms, dt_ms):
        self.t = np.arange(0, t_sim_ms, dt_ms)
        self.total_energy = 1e-18 # Joules (willkürlich)

    def get_coherent_burst(self, t_center_ms, duration_ms=0.1):
        """Simuliert einen scharfen, superradianten Puls."""
        pulse = np.zeros_like(self.t)
        mask = (self.t > t_center_ms) & (self.t < t_center_ms + duration_ms)
        if np.any(mask):
            power = self.total_energy / (duration_ms * 1e-3)
            pulse[mask] = power
        return pulse

    def get_incoherent_noise(self, t_center_ms, duration_ms=10.0):
        """Simuliert einen schwachen, verrauschten Puls gleicher Energie."""
        pulse = np.zeros_like(self.t)
        mask = (self.t > t_center_ms) & (self.t < t_center_ms + duration_ms)
        if np.any(mask):
            power = self.total_energy / (duration_ms * 1e-3)
            # Gaußsches Rauschen um den Mittelwert
            pulse[mask] = np.random.normal(power, power * 0.3, size=np.sum(mask))
            pulse[pulse < 0] = 0
        return pulse

class NeuralNetwork:
    """Ein Netzwerk aus Leaky Integrate-and-Fire (LIF) Neuronen mit STDP."""

    def __init__(self, n_input, n_engram, seed=42):
        self.rng = np.random.default_rng(seed)
        self.n_input = n_input
        self.n_engram = n_engram

        # LIF-Parameter
        self.V_rest = -70.0; self.V_th = -55.0; self.V_reset = -75.0
        self.tau_m = 10.0 # ms
        self.R_m = 10.0 # MΩ

        # STDP-Parameter
        self.eta_ltp = 0.005  # Lernrate
        self.tau_stdp = 5.0 # ms Zeitfenster für Plastizität

        # Synaptische Gewichte [von, zu] -> [input, engram]
        self.W = self.rng.uniform(0.0, 0.5, size=(self.n_engram, self.n_input))
        self.W_initial = self.W.copy()

    def run(self, t_sim_ms, dt_ms, input_current, photonic_power=None):
        """Simuliert das Netzwerk für eine gegebene Dauer."""
        time = np.arange(0, t_sim_ms, dt_ms)
        V = np.full(self.n_engram, self.V_rest)
        last_spike_input = np.full(self.n_input, -np.inf)
        last_spike_engram = np.full(self.n_engram, -np.inf)

        spikes_engram = [] # Liste von (time, neuron_idx)

        # Photonische Kopplung (vereinfacht: Power -> Strom)
        photonic_coupling = 1e11 # A/W

        for i, t in enumerate(time):
            # Input von externem Muster
            I_syn = self.W @ input_current[:, i]

            # Input von Photonen (globales Signal an alle Engram-Neuronen)
            I_photonic = 0
            if photonic_power is not None:
                I_photonic = photonic_power[i] * photonic_coupling

            # LIF-Dynamik
            dV = (-(V - self.V_rest) + self.R_m * (I_syn + I_photonic)) / self.tau_m
            V += dV * dt_ms

            # Spikes
            spiked_neurons = np.where(V >= self.V_th)[0]

            if len(spiked_neurons) > 0:
                V[spiked_neurons] = self.V_reset
                for neuron_idx in spiked_neurons:
                    spikes_engram.append((t, neuron_idx))
                    last_spike_engram[neuron_idx] = t

                    # STDP Update: Post-synaptischer Spike -> LTP
                    # Finde alle Input-Neuronen, die kurz zuvor gefeuert haben
                    active_inputs = np.where(input_current[:, i] > 0)[0]
                    for input_idx in active_inputs:
                        if (t - last_spike_input[input_idx]) < self.tau_stdp:
                            # Hebb'sche Regel: Verstärke die Synapse
                            dw = self.eta_ltp * (1.0 - self.W[neuron_idx, input_idx])
                            self.W[neuron_idx, input_idx] += dw

            # Update der letzten Spike-Zeiten für den Input
            active_inputs = np.where(input_current[:, i] > 0)[0]
            last_spike_input[active_inputs] = t

        return spikes_engram

# =============================================================================
# 2. Experiment-Protokoll
# =============================================================================

def create_input_pattern(t, n_input, active_indices, t_start_ms, t_end_ms, rate_hz):
    """Erzeugt einen Spike-Train für eine Gruppe von Neuronen."""
    current = np.zeros((n_input, len(t)))
    dt_ms = t[1] - t[0]
    prob = rate_hz * (dt_ms / 1000.0)

    time_mask = (t >= t_start_ms) & (t < t_end_ms)

    for i in active_indices:
        spikes = np.random.rand(np.sum(time_mask)) < prob
        current[i, time_mask] = spikes * 1.5 # nA
    return current


def plot_results(results: dict):
    """Optionales Plotten der Ergebnisse."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("Emergence of an Associative Memory Engram", fontsize=16)

    # Gewichtsmatrizen
    im1 = axs[0, 0].imshow(results["W_initial"], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[0, 0].set_title("Initial Synaptic Weights")
    axs[0, 0].set_ylabel("Engram Neurons")
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(results["W_hyp_trained"], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[0, 1].set_title("Weights after HYPOTHESIS Training")
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].imshow(results["W_ant_trained"], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[1, 0].set_title("Weights after ANTITHESIS Training")
    axs[1, 0].set_xlabel("Input Neurons")
    axs[1, 0].set_ylabel("Engram Neurons")
    fig.colorbar(im3, ax=axs[1, 0])

    # Text-Zusammenfassung
    axs[1, 1].axis('off')
    summary_text = (
        f"Selectivity Score (HYP): {results['selectivity_hyp']:.2f}\n"
        f"Selectivity Score (ANT): {results['selectivity_ant']:.2f}\n\n"
        f"Pattern Completion (HYP): {results['completion_hyp']:.1f} spikes\n"
        f"Pattern Completion (ANT): {results['completion_ant']:.1f} spikes"
    )
    axs[1, 1].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = "neuro_emergent_system_results.png"
    plt.savefig(filename)
    print(f"\n[OK] Plot gespeichert: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Stufe 3: Simulation emergenter Neuro-Eigenschaften.")
    parser.add_argument("--plots", action="store_true", help="Generiere PNG-Plots der Ergebnisse.")
    parser.add_argument("--seed", type=int, default=42, help="Seed für Zufallsgeneratoren.")
    args = parser.parse_args()

    # --- Setup ---
    N_INPUT = 50
    N_ENGRAM = 20
    T_SIM_MS = 100.0
    DT_MS = 0.1

    photonic_sys = PhotonicSystem(T_SIM_MS, DT_MS)
    time_vec = np.arange(0, T_SIM_MS, DT_MS)

    # Definiere Muster
    pattern_A_indices = np.arange(0, N_INPUT // 2)
    pattern_B_indices = np.arange(N_INPUT // 2, N_INPUT)
    pattern_A_noisy_indices = np.random.choice(pattern_A_indices, size=len(pattern_A_indices)//2, replace=False)

    input_A = create_input_pattern(time_vec, N_INPUT, pattern_A_indices, 20, 80, 50)
    input_B = create_input_pattern(time_vec, N_INPUT, pattern_B_indices, 20, 80, 50)
    input_A_noisy = create_input_pattern(time_vec, N_INPUT, pattern_A_noisy_indices, 20, 80, 50)

    # Zwei separate Netzwerk-Instanzen für die beiden Trainings-Bedingungen
    net_hyp = NeuralNetwork(N_INPUT, N_ENGRAM, seed=args.seed)
    net_ant = NeuralNetwork(N_INPUT, N_ENGRAM, seed=args.seed)

    results = {"W_initial": net_hyp.W_initial.copy()}

    print("\n" + "="*50)
    print("        PHASE 1: PRE-TRAINING (NAIVES NETZWERK)")
    print("="*50)

    spikes_pre_A = net_hyp.run(T_SIM_MS, DT_MS, input_A)
    spikes_pre_B = net_hyp.run(T_SIM_MS, DT_MS, input_B)
    print(f"Antwort auf Pattern A (vorher): {len(spikes_pre_A)} Spikes")
    print(f"Antwort auf Pattern B (vorher): {len(spikes_pre_B)} Spikes")

    print("\n" + "="*50)
    print("        PHASE 2: TRAINING")
    print("="*50)

    # HYPOTHESE: Training mit kohärentem Burst
    print("Trainiere HYP-Netzwerk (Pattern A + kohärenter SR-Burst)...")
    sr_burst = photonic_sys.get_coherent_burst(t_center_ms=50)
    _ = net_hyp.run(T_SIM_MS, DT_MS, input_A, photonic_power=sr_burst)
    results["W_hyp_trained"] = net_hyp.W.copy()

    # ANTITHESE: Training mit inkohärentem Rauschen
    print("Trainiere ANT-Netzwerk (Pattern A + inkohärentes Rauschen)...")
    incoherent_pulse = photonic_sys.get_incoherent_noise(t_center_ms=45)
    _ = net_ant.run(T_SIM_MS, DT_MS, input_A, photonic_power=incoherent_pulse)
    results["W_ant_trained"] = net_ant.W.copy()

    print("\n" + "="*50)
    print("        PHASE 3: POST-TRAINING (TEST AUF GEDÄCHTNIS)")
    print("="*50)

    # Test HYP-Netz
    spikes_post_A_hyp = net_hyp.run(T_SIM_MS, DT_MS, input_A)
    spikes_post_B_hyp = net_hyp.run(T_SIM_MS, DT_MS, input_B)
    spikes_post_A_noisy_hyp = net_hyp.run(T_SIM_MS, DT_MS, input_A_noisy)

    # Test ANT-Netz
    spikes_post_A_ant = net_ant.run(T_SIM_MS, DT_MS, input_A)
    spikes_post_B_ant = net_ant.run(T_SIM_MS, DT_MS, input_B)
    spikes_post_A_noisy_ant = net_ant.run(T_SIM_MS, DT_MS, input_A_noisy)

    # --- Auswertung ---
    resp_A_hyp = len(spikes_post_A_hyp)
    resp_B_hyp = len(spikes_post_B_hyp)
    results["selectivity_hyp"] = (resp_A_hyp - resp_B_hyp) / (resp_A_hyp + resp_B_hyp + 1e-9)
    results["completion_hyp"] = len(spikes_post_A_noisy_hyp)

    resp_A_ant = len(spikes_post_A_ant)
    resp_B_ant = len(spikes_post_B_ant)
    results["selectivity_ant"] = (resp_A_ant - resp_B_ant) / (resp_A_ant + resp_B_ant + 1e-9)
    results["completion_ant"] = len(spikes_post_A_noisy_ant)

    print("\n--- ERGEBNISSE HYPOTHESE-NETZWERK ---")
    print(f"Antwort auf gelerntes Pattern A: {resp_A_hyp} Spikes")
    print(f"Antwort auf neues Pattern B:     {resp_B_hyp} Spikes")
    print(f"--> Selektivitäts-Score: {results['selectivity_hyp']:.2f} (1.0 = perfekt)")
    print(f"Antwort auf unvollständiges A (Pattern Completion): {results['completion_hyp']} Spikes")

    print("\n--- ERGEBNISSE ANTITHESE-NETZWERK ---")
    print(f"Antwort auf gelerntes Pattern A: {resp_A_ant} Spikes")
    print(f"Antwort auf neues Pattern B:     {resp_B_ant} Spikes")
    print(f"--> Selektivitäts-Score: {results['selectivity_ant']:.2f} (nahe 0 = kein Lernen)")
    print(f"Antwort auf unvollständiges A (Pattern Completion): {results['completion_ant']} Spikes")

    print("\n" + "="*50)
    print("        SCHLUSSFOLGERUNG")
    print("="*50)
    if results['selectivity_hyp'] > 0.8 and results['selectivity_ant'] < 0.2:
        print("Die Simulation stützt die Hypothese: Ein kohärenter Photonen-Burst ermöglichte die")
        print("Ausbildung eines selektiven neuronalen Engrams. Ohne diese präzise Synchronisation")
        print("fand kein signifikantes Lernen statt. Die Fähigkeit zur Mustererkennung ist hier")
        print("ein emergenter 'sinnvoller Ausdruck', der kausal von der Kohärenz des Lichtsignals abhängt.")
        print("Die Antithese (Photonen sind nur Rauschen) wurde in diesem Modell falsifiziert.")
    else:
        print("Die Simulation konnte die Hypothese nicht eindeutig stützen. Die Selektivität war")
        print("in beiden Bedingungen gering oder der Unterschied nicht signifikant.")

    if args.plots:
        plot_results(results)

if __name__ == "__main__":
    main()
