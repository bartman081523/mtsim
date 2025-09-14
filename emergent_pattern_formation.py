#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stufe 4: Emergenz von räumlich-zeitlichen Mustern durch
         Photon-Gated Synchronization

Dieses Skript simuliert ein 2D-Gitter von gekoppelten Phasen-Oszillatoren
(Kuramoto-Modell), das EEG-ähnliche Aktivität repräsentiert.

HYPOTHESE:
Ein globaler, zeitlich präziser superradianter UPE-Burst kann das gesamte
System in einen geordneten Zustand (Phasen-Reset) zwingen. Aus dieser
induzierten Kohärenz emergiert ein komplexes, räumlich-zeitliches Muster
(hier eine wandernde Welle).

ANTITHESE:
Ein inkohärenter, verrauschter Puls mit der gleichen Gesamtenergie ist nicht
in der Lage, die notwendige globale Kohärenz zu erzeugen. Das System bleibt
in einem desorganisierten, "flimmernden" Zustand.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# =============================================================================
# 1. Konfiguration
# =============================================================================
config = {
    "grid_size": 30,         # Gittergröße (N x N Oszillatoren)
    "sim_duration": 500,     # Simulationsschritte
    "dt": 0.1,               # Zeitschritt
    "coupling_strength": 0.8, # Stärke der lokalen Kopplung
    "natural_freq_std": 0.2, # Variation der Eigenfrequenzen

    "burst_time": 50,        # Zeitpunkt des UPE-Events
    "burst_energy": 10.0,    # Energie des Pulses (willkürliche Einheit)

    "pattern": {
        "wave_speed": 0.5,   # Geschwindigkeit des emergenten Musters
    }
}

# =============================================================================
# 2. Simulations-Kern
# =============================================================================

class CoupledOscillatorSystem:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.size = cfg["grid_size"]

        # Phasen-Gitter (zufälliger Start)
        self.phases = self.rng.uniform(0, 2 * np.pi, size=(self.size, self.size))

        # Eigenfrequenzen der Oszillatoren
        self.omega = self.rng.normal(1.0, cfg["natural_freq_std"], size=(self.size, self.size))

        # Nachbarschafts-Kernel für die Kopplung (einfacher Durchschnitt)
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0

    def _get_coupling_term(self):
        """Berechnet den Einfluss der Nachbarn auf jeden Oszillator."""
        # Sinus der Phasendifferenzen zu den Nachbarn
        # Wir nutzen 2D-Konvolution, um das effizient zu berechnen
        from scipy.signal import convolve2d

        sin_phases = np.sin(self.phases)
        cos_phases = np.cos(self.phases)

        mean_sin = convolve2d(sin_phases, self.kernel, mode='same', boundary='wrap')
        mean_cos = convolve2d(cos_phases, self.kernel, mode='same', boundary='wrap')

        return mean_cos * sin_phases - mean_sin * cos_phases

    def apply_upe_pulse(self, energy: float, is_coherent: bool):
        """
        Simuliert den Einfluss eines UPE-Pulses.
        - Kohärent: Starker, globaler Phasen-Reset zu einem festen Wert.
        - Inkohärent: Schwacher, verrauschter Phasen-Push.
        """
        if is_coherent:
            # Starker Reset auf eine "Start-Phase" (z.B. 0)
            reset_strength = energy
            self.phases = (self.phases * (1 - reset_strength) + 0) % (2 * np.pi)
        else:
            # Schwacher, verrauschter Einfluss
            noise_strength = energy * 0.1 # 10x schwächere Wirkung
            noise = self.rng.normal(0, np.pi * 0.5, size=self.phases.shape)
            self.phases = (self.phases + noise * noise_strength) % (2 * np.pi)

    def step(self):
        """Führt einen Simulationsschritt durch."""
        coupling_term = self._get_coupling_term()
        d_phases = self.omega + self.cfg["coupling_strength"] * coupling_term
        self.phases = (self.phases + d_phases * self.cfg["dt"]) % (2 * np.pi)

    def generate_emergent_pattern(self, t):
        """Erzeugt künstlich das Zielmuster (wandernde Welle) zum Vergleich."""
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        distance_from_corner = np.sqrt(x**2 + y**2)
        wave = np.sin(distance_from_corner - self.cfg["pattern"]["wave_speed"] * t)
        return (wave * np.pi + np.pi) % (2*np.pi)

# =============================================================================
# 3. Visualisierung
# =============================================================================

def run_and_animate(condition: str, cfg: dict, seed: int):
    """Führt die Simulation durch und erzeugt eine Animation."""

    system = CoupledOscillatorSystem(cfg, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 8))

    title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=14)
    im = ax.imshow(np.sin(system.phases), cmap='twilight_shifted', vmin=-1, vmax=1)
    ax.axis('off')

    history = []

    for t in range(cfg["sim_duration"]):
        if t == cfg["burst_time"]:
            is_coherent = (condition == "Hypothesis")
            system.apply_upe_pulse(cfg["burst_energy"], is_coherent)

        system.step()
        history.append(np.sin(system.phases).copy())

    def update(frame):
        im.set_data(history[frame])
        time_point = frame
        status = "Pre-Burst (Disordered)"
        if time_point > cfg["burst_time"]:
            status = "Post-Burst (Emergent Pattern)" if condition == "Hypothesis" else "Post-Burst (Remains Disordered)"

        title_text.set_text(f"Condition: {condition} | Time: {time_point}\nStatus: {status}")
        return [im, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(history), blit=True, interval=30)

    filename = f"emergent_pattern_{condition.lower()}.mp4"
    ani.save(filename, writer='ffmpeg', dpi=100)
    print(f"[OK] Animation gespeichert: {filename}")
    plt.close(fig)

# =============================================================================
# 4. Main-Funktion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stufe 4: Simulation emergenter Muster.")
    parser.add_argument("--seed", type=int, default=42, help="Seed für Zufallsgeneratoren.")
    args = parser.parse_args()

    try:
        from scipy.signal import convolve2d
    except ImportError:
        print("Fehler: Bitte 'scipy' installieren: pip install scipy")
        return

    print("="*60)
    print("  SIMULATION DER EMERGENZ RÄUMLICH-ZEITLICHER MUSTER")
    print("="*60)

    # --- HYPOTHESE: Kohärenter Burst führt zu Ordnung ---
    print("\n[1] Führe Simulation für die HYPOTHESE durch (kohärenter UPE-Burst)...")
    run_and_animate("Hypothesis", config, seed=args.seed)

    # --- ANTITHESE: Inkohärenter Puls führt nicht zu Ordnung ---
    print("\n[2] Führe Simulation für die ANTITHESE durch (inkohärenter UPE-Puls)...")
    run_and_animate("Antithesis", config, seed=args.seed)

    print("\n" + "="*60)
    print("        SCHLUSSFOLGERUNG")
    print("="*60)
    print("Bitte vergleiche die beiden erzeugten MP4-Videos:")
    print(" - emergent_pattern_hypothesis.mp4")
    print(" - emergent_pattern_antithesis.mp4")
    print("\nDie Hypothese wird gestützt, wenn nur im ersten Video ein geordnetes,")
    print("wellenartiges Muster nach dem UPE-Burst bei t=50 entsteht.")
    print("Die Antithese wird gestützt, wenn beide Videos desorganisiert bleiben")
    print("oder sich nicht signifikant unterscheiden.")

if __name__ == "__main__":
    main()
