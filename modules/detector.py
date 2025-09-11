# file: modules/detector.py
import numpy as np

class Detector:
    def __init__(self, QE=0.6, eta_geom=0.1, mu_eff_mm=10, r_um=50, dark=100, window=1e-7):
        self.QE=QE; self.eta=eta_geom
        self.mu=mu_eff_mm*1000.0; self.r=r_um*1e-6
        self.dark=dark; self.win=window
    def measure(self, N_emit):
        atten=np.exp(-self.mu*self.r)
        N_det=N_emit*atten*self.eta*self.QE
        snr = N_det/np.sqrt(N_det + self.dark*self.win)
        return float(N_det), float(snr)
