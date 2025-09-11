# file: run_hybrid_sim.py
import numpy as np
from pathlib import Path

# --- Imports aus externen Repos (leichte Adapter liegen in ./adapters)
from adapters.mt_spindle import MTSpindleSystem
from adapters.kmc_motors import MotorLattice
from modules.photonic import PhotonicEmitter
from modules.spin_ros import SpinROS
from modules.detector import Detector

def run_scenario(curvature, mode, loss_mm=10, r_um=50):
    # 1) Zellsystem (MT + Spindel)
    mt = MTSpindleSystem(n_mt=9, length_um=3.0)     # aus Lera-Ramírez inspiriert
    motors = MotorLattice(mt_graph=mt.graph)        # aus kMC_MoTub inspiriert

    # 2) Photonik/Spin-Module
    phot = PhotonicEmitter(cluster_size=50, n_clusters=20, mode=mode, curvature=curvature)
    spin  = SpinROS(curvature=curvature)

    # 3) Simulationsschritte
    dt = 1e-12; T = 5e-9; time = np.arange(0, T, dt)
    I_t = phot.emit(time)                            # UPE-Zeitverlauf (incoh vs SR)
    motors.step_series(I_t)                          # Kopplung: Photonfeld -> Motorbias
    mt.update_dynamics(dt_series=time*0+dt)          # ggf. DI-Raten (portiert aus 4-state)

    # 4) Detektor/SNR
    det = Detector(QE=0.6, eta_geom=0.1, mu_eff_mm=loss_mm, r_um=r_um, dark=100, window=1e-7)
    N_emit = phot.total_photons(I_t, time)
    N_det, snr = det.measure(N_emit)

    # 5) Outputs
    return {
        "mode": mode,
        "curvature": curvature,
        "loss_mm": loss_mm,
        "distance_um": r_um,
        "emit_photons": float(N_emit),
        "det_photons": float(N_det),
        "snr": float(snr),
        "motor_net_steps": int(motors.net_displacement()),
        "photon_peak": float(I_t.max()),
        "cilia_phase_reset": float(phot.phase_reset(I_t)),
        "spindle_metric": float(mt.spindle_metric(spin.modulator()))
    }

if __name__ == "__main__":
    for mode in ["hypothesis","antithesis"]:
        for kappa in [0.0,0.5,1.0]:
            res = run_scenario(curvature=kappa, mode=mode, loss_mm=10, r_um=50)
            print(res)
