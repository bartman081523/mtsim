#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Callable, Iterable, Tuple, Dict
import numpy as np

# ----------- Physikalische Konstanten -----------
HBAR = 1.054e-34
C = 3e8
LAM = 280e-9               # 280 nm (UV)
OMEGA = 2*np.pi*C/LAM
E_PHOTON = HBAR*OMEGA
TAU_SP = 1e-9              # Einzel-Emitter-Lebensdauer

# ----------- Zeitachsen-Defaults -----------
DT_DEFAULT = 1e-12         # 1 ps
T_DEFAULT  = 5e-9          # 5 ns

# =================== Hilfsmetriken ===================

def peak_sharpness(I: np.ndarray) -> float:
    m = float(I.mean()) if I.size else 0.0
    return float(I.max() / (m + 1e-30))

def peak_to_rms(I: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(I**2))) if I.size else 0.0
    return float(I.max() / (rms + 1e-30))

# =================== Photonik ===================

def I_incoherent(N_emit: int, t: np.ndarray, tau: float = TAU_SP) -> np.ndarray:
    # Breiter exponentieller Zerfall
    return (N_emit * E_PHOTON / max(tau, 1e-30)) * np.exp(-t/max(tau, 1e-30))

def I_superradiant(N_emit: int, t: np.ndarray, tau: float = TAU_SP) -> np.ndarray:
    # Dicke-ähnlicher sech^2-Puls
    Np = max(N_emit, 1)
    tau_sr = tau / Np
    t_d = tau_sr * np.log(Np)
    return (N_emit*E_PHOTON/max(tau,1e-30)) * (N_emit+1)/4.0 * (1/np.cosh((t - t_d)/max(tau_sr,1e-30)))**2

class PhotonicEmitter:
    """
    mode:
      - "hypothesis" -> superradiant (enge zeitliche Koordination)
      - "antithesis" -> inkohärent (breiter Jitter & Tau-Dispersion)
    curvature (κ) modifiziert Gain (heuristisch).
    emission_scale skaliert Gesamtintensität.
    Jitter/Dispersion sind entscheidend zur Diskriminierung.
    """
    def __init__(self, cluster_size=50, n_clusters=20, mode="hypothesis",
                 curvature: float = 0.0, emission_scale: float = 1.0,
                 sigma_t_hyp: float = 5e-12,   # 5 ps
                 sigma_t_ant: float = 5e-10,   # 0.5 ns
                 tau_dispersion_ant: float = 0.25,
                 rng: Optional[np.random.Generator] = None):
        self.N = int(cluster_size)
        self.M = int(n_clusters)
        self.mode = mode
        self.kappa = float(curvature)
        self.es  = float(emission_scale)
        self.sigma_t_hyp = float(sigma_t_hyp)
        self.sigma_t_ant = float(sigma_t_ant)
        self.tau_disp_ant = float(tau_dispersion_ant)
        self.rng = np.random.default_rng() if rng is None else rng

    def emit(self, t: np.ndarray) -> np.ndarray:
        I = np.zeros_like(t, dtype=float)
        r = self.rng

        if self.mode == "hypothesis":
            # enger Jitter
            for _ in range(self.M):
                t0 = r.normal(0.0, self.sigma_t_hyp)
                I += I_superradiant(self.N, t - t0)
            gain = (1 + 0.5*self.kappa)
        else:
            # breiter Jitter + Tau-Dispersion
            for _ in range(self.M):
                t0 = r.normal(0.0, self.sigma_t_ant)
                tau_i = TAU_SP * float(r.lognormal(mean=0.0, sigma=self.tau_disp_ant))
                I += I_incoherent(self.N, t - t0, tau=tau_i)
            gain = 1/(1 + 2*self.kappa)

        return I * gain * self.es

    @staticmethod
    def total_photons(I: np.ndarray, t: np.ndarray) -> float:
        return float(np.trapezoid(I, t) / E_PHOTON)

    @staticmethod
    def phase_reset(I: np.ndarray) -> float:
        return 5e-3 * (float(I.max()) / (float(I.mean()) + 1e-30))


def photons_in_time_gate(I: np.ndarray, t: np.ndarray, window_s: float) -> float:
    """
    Integriert I(t) nur in einem Zeitfenster (Breite = window_s) um das Maximum von I(t).
    Gibt die Photonenzahl im Gate zurück.
    """
    if window_s <= 0:
        return 0.0
    idx_peak = int(np.argmax(I))
    t0 = t[idx_peak] - 0.5*window_s
    t1 = t[idx_peak] + 0.5*window_s
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        mask = np.zeros_like(t, dtype=bool); mask[idx_peak] = True
    E_gate = float(np.trapezoid(I[mask], t[mask]))
    return E_gate / E_PHOTON

def photons_in_time_gate_window(I: np.ndarray, t: np.ndarray, t0: float, t1: float) -> float:
    """Photonen im festen Zeitfenster [t0, t1] (globales Gate)."""
    if t1 <= t0:
        return 0.0
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return 0.0
    E_gate = float(np.trapezoid(I[mask], t[mask]))
    return E_gate / E_PHOTON


# =================== Spin/ROS ===================

GAMMA_E = 1.76e11
class SpinROS:
    def __init__(self, curvature: float = 0.0, T2_base: float = 1e-9):
        self.T2 = T2_base * (1 + float(curvature))
    def modulator(self, B: float = 1e-6, t: float = 1e-9) -> float:
        return 0.5 * (1 + np.cos(GAMMA_E*B*t) * np.exp(-t/self.T2))


# =================== 4-State MT/Spindel ===================

class FourStateMT:
    GROW, SHRINK, PAUSE = 0, 1, 2
    def __init__(self, vg_nm_s=300, vs_nm_s=600, k_cat=0.2, k_res=0.05, k_pause=0.05, k_unpause=0.2,
                 length_nm_init=2000, rng: Optional[np.random.Generator]=None):
        self.vg = vg_nm_s; self.vs = vs_nm_s
        self.k_cat = k_cat; self.k_res = k_res
        self.k_pause = k_pause; self.k_unpause = k_unpause
        self.L = float(length_nm_init); self.state = self.GROW
        self.rng = np.random.default_rng() if rng is None else rng
    def step(self, dt_s: float):
        r = self.rng
        if self.state == self.GROW:
            if r.random() < 1 - np.exp(-self.k_cat*dt_s): self.state = self.SHRINK
            elif r.random() < 1 - np.exp(-self.k_pause*dt_s): self.state = self.PAUSE
        elif self.state == self.SHRINK:
            if r.random() < 1 - np.exp(-self.k_res*dt_s): self.state = self.GROW
            elif r.random() < 1 - np.exp(-self.k_pause*dt_s): self.state = self.PAUSE
        elif self.state == self.PAUSE:
            if r.random() < 1 - np.exp(-self.k_unpause*dt_s): self.state = self.GROW if r.random() < 0.7 else self.SHRINK
        if self.state == self.GROW: self.L += self.vg * dt_s
        elif self.state == self.SHRINK: self.L = max(0.0, self.L - self.vs * dt_s)

class MTSpindleSystem:
    def __init__(self, n_mt=9, length_um=3.0, seed=1234):
        self.rng = np.random.default_rng(seed)
        target_nm = length_um*1000.0
        self.mts = [FourStateMT(length_nm_init=target_nm/2, rng=self.rng) for _ in range(n_mt)]
        self._last_sum = self.sum_len()
    def sum_len(self) -> float: return float(sum(mt.L for mt in self.mts))
    def update_dynamics(self, dt_series_s: np.ndarray, ros_modulator: float = 1.0):
        for mt in self.mts:
            base_kcat, base_kres = mt.k_cat, mt.k_res
            mt.k_cat = base_kcat / max(ros_modulator, 1e-30)
            mt.k_res = base_kres * ros_modulator
        for dt in dt_series_s:
            for mt in self.mts: mt.step(dt)
    def spindle_metric(self, ros_modulator: float = 1.0) -> float:
        total = self.sum_len()
        dL = (total - self._last_sum) * ros_modulator
        self._last_sum = total
        return float(dL)


# =================== Motoren (Proxy) ===================

class MotorLattice:
    def __init__(self): self.pos = 0
    def step_series(self, I_t: np.ndarray):
        Imax = max(float(I_t.max()), 1e-30)
        for It in I_t:
            p_fwd = 0.5 + 0.1*(float(It)/Imax)
            p_fwd = min(max(p_fwd, 0.0), 1.0)
            self.pos += 1 if np.random.rand() < p_fwd else -1
    def net_displacement(self) -> int: return int(self.pos)


# =================== Detektor ===================

class Detector:
    def __init__(self, QE=0.6, eta_geom=0.1, mu_eff_mm=10.0, dark=100.0, window=1e-7):
        self.QE = float(QE)
        self.eta = float(eta_geom)
        self.mu = float(mu_eff_mm) * 1000.0  # 1/m
        self.dark = float(dark)              # counts/s
        self.win  = float(window)            # s

    def measure_counts_distance(self, N_emit_gate: float, r_um: float) -> float:
        r_m = float(r_um) * 1e-6
        atten = np.exp(-self.mu * r_m)
        return float(N_emit_gate * atten * self.eta * self.QE)

    @staticmethod
    def apply_deadtime_afterpulse(N_det: float, window_s: float, deadtime_s: float, afterpulse: float) -> float:
        if window_s > 0 and deadtime_s > 0:
            N_det = N_det / (1.0 + (N_det * deadtime_s / window_s))
        if afterpulse > 0:
            N_det = N_det * (1.0 + afterpulse)
        return N_det

    def snr_from_counts(self, N_det: float, afterpulse: float = 0.0) -> float:
        var_eff = N_det + self.dark*self.win + afterpulse*max(N_det, 0.0)
        return float(N_det / np.sqrt(max(var_eff, 1e-30)))


# =================== Hybrid-Einzelsimulation ===================

def simulate_hybrid_once(
    curvature: float, mode: str, loss_mm: float, r_um: float, seed: int,
    window_s: float, deadtime_s: float, afterpulse: float,
    qe: float, eta_geom: float, dark_cps: float, emission_scale: float,
    poissonize: bool = False,
    dt: float = DT_DEFAULT, T: float = T_DEFAULT
) -> Dict[str, float]:
    np.random.seed(seed)
    time = np.arange(0, T, dt)

    phot = PhotonicEmitter(cluster_size=50, n_clusters=20, mode=mode,
                           curvature=curvature, emission_scale=emission_scale)
    spin = SpinROS(curvature=curvature)
    mt = MTSpindleSystem(n_mt=9, length_um=3.0, seed=seed)
    motors = MotorLattice()

    I_t = phot.emit(time)
    motors.step_series(I_t)
    mt.update_dynamics(dt_series_s=time*0+dt, ros_modulator=spin.modulator())

    det = Detector(QE=qe, eta_geom=eta_geom, mu_eff_mm=loss_mm, dark=dark_cps, window=window_s)

    # Diagnose: volle Emission + im Gate
    N_emit_total = PhotonicEmitter.total_photons(I_t, time)
    N_emit_gate  = photons_in_time_gate(I_t, time, window_s)
    coincidence_ratio = float(N_emit_gate / (N_emit_total + 1e-30))
    sharp = peak_sharpness(I_t)
    p2r = peak_to_rms(I_t)

    # Detektion (nur Gate-Photonen)
    N_det = det.measure_counts_distance(N_emit_gate, r_um)
    N_det_eff = Detector.apply_deadtime_afterpulse(N_det, window_s, deadtime_s, afterpulse)

    # nach Poisson-Sampling:
    if poissonize and window_s > 0:
        lam_sig = max(N_det_eff, 0.0)         # erwartete Signalcounts im Gate
        lam_dark = det.dark*window_s          # erwartete Darks im Gate
        k = np.random.poisson(lam_sig + lam_dark)
        # Schätzer für Signal:
        N_det_eff = max(float(k) - lam_dark, 0.0)
        # >>> NEU: SNR auf Basis erwarteter Varianz (Signal+Dark), nicht zentriertes Sample:
        snr = float(lam_sig / np.sqrt(lam_sig + lam_dark + max(afterpulse*lam_sig,0.0)))
    else:
        snr = det.snr_from_counts(N_det_eff, afterpulse)

    # --- Kuramoto/PRC auf Hybrid-Level (synthetische Oszillatorwolke) ---
    # Drive aus Wellenform:
    Imax = float(I_t.max())
    Irms = float(np.sqrt(np.mean(I_t**2))) + 1e-30
    drive = Imax / Irms  # ~peak_sharpness

    Nosc = 16
    rng = np.random.default_rng(seed + 9999)
    phi0 = rng.uniform(0, 2*np.pi, size=Nosc)

    def _kuramoto_R(phi):
        z = np.exp(1j*phi)
        return float(np.abs(np.mean(z)))

    # PRC: Δφ = eps * sin(φ) * drive
    eps = 0.2
    dphi = eps * np.sin(phi0) * drive
    phi1 = (phi0 + dphi) % (2*np.pi)

    R_before = _kuramoto_R(phi0)
    R_after  = _kuramoto_R(phi1)
    sync_gain = R_after - R_before
    mean_phase_reset = float(np.mean(np.abs(dphi)))

    # Score (wird von den Probe-/Dynamics-Skripten erwartet)
    score = float(p2r * coincidence_ratio)

    return {
        "mode": mode,
        "curvature": float(curvature),
        "loss_mm": float(loss_mm),
        "distance_um": float(r_um),

        "emit_photons": float(N_emit_total),   # Referenz: total
        "gate_photons": float(N_emit_gate),    # Photonen im Gate
        "coincidence_ratio": float(coincidence_ratio),
        "peak_sharpness": float(sharp),
        "p2rms": float(p2r),
        "score": float(score),

        "det_photons": float(N_det_eff),
        "snr": float(snr),

        "R_before": float(R_before),
        "R_after": float(R_after),
        "sync_gain": float(sync_gain),
        "mean_phase_reset": float(mean_phase_reset),

        "motor_net_steps": motors.net_displacement(),
        "photon_peak": float(I_t.max()),
        "cilia_phase_reset": float(PhotonicEmitter.phase_reset(I_t)),
        "spindle_metric": float(mt.spindle_metric(spin.modulator()))
    }

def simulate_hybrid_avg(
    curvature: float, mode: str, loss_mm: float, r_um: float, trials: int, base_seed: int,
    window_s: float, deadtime_s: float, afterpulse: float,
    qe: float, eta_geom: float, dark_cps: float, emission_scale: float,
    poissonize: bool = False,
    pbar: Optional[Callable[[Iterable], Iterable]] = None
) -> Dict[str, float]:
    assert trials > 0
    rows=[]; it=range(trials)
    if pbar is not None:
        it = pbar(it, total=trials, leave=False,
                  desc=f"Trials {mode}, κ={curvature}, μ={loss_mm}, r={r_um}µm")
    for i in it:
        seed = base_seed + i
        rows.append(simulate_hybrid_once(
            curvature, mode, loss_mm, r_um, seed, window_s, deadtime_s, afterpulse,
            qe, eta_geom, dark_cps, emission_scale, poissonize=poissonize
        ))
    meta = {"mode","curvature","loss_mm","distance_um"}
    out = {"trials": trials}
    for k in meta: out[k] = rows[0][k]
    for k,v in rows[0].items():
        if k in meta or not isinstance(v,(int,float)): continue
        arr = np.array([r[k] for r in rows], dtype=float)
        out[k+"_mean"] = float(arr.mean())
        out[k+"_sem"]  = float(arr.std(ddof=1)/np.sqrt(trials)) if trials>1 else 0.0
    return out


# =================== Netzwerk-Tools (für Vollständigkeit; hier ungenutzt) ===================

def place_nodes(N:int, radius_um: float, seed:int) -> np.ndarray:
    rng = np.random.default_rng(seed); pts=[]
    while len(pts)<N:
        p = rng.uniform(-radius_um, radius_um, size=3)
        if np.linalg.norm(p) <= radius_um: pts.append(p)
    return np.array(pts)

def pairwise_dist_um(P: np.ndarray) -> np.ndarray:
    diff = P[:,None,:]-P[None,:,:]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, 0.0)
    return D

def coupling_matrix(D_um: np.ndarray, ell_um: float, alpha: float) -> np.ndarray:
    if ell_um <= 0: return np.zeros_like(D_um)
    K = alpha * np.exp(-D_um/ell_um)
    np.fill_diagonal(K, 0.0)
    return K

def prc_delta_phi(phi: np.ndarray, Ipeak_norm: np.ndarray, eps: float = 0.2, phi0: float = 0.0) -> np.ndarray:
    return eps*np.sin(phi - phi0) * Ipeak_norm

def kuramoto_R(phi: np.ndarray) -> float:
    z = np.exp(1j*phi)
    return float(np.abs(np.mean(z)))
