# file: modules/photonic.py
import numpy as np

hbar=1.054e-34; c=3e8; lam=280e-9; omega=2*np.pi*c/lam; E=hbar*omega; tau=1e-9

def I_incoh(N,t): return (N*E/tau)*np.exp(-t/tau)
def I_SR(N,t):
    tau_SR=tau/N; t_d=tau_SR*np.log(N)
    return (N*E/tau)*(N+1)/4*(1/np.cosh((t-t_d)/tau_SR))**2

class PhotonicEmitter:
    def __init__(self, cluster_size=50, n_clusters=20, mode="hypothesis", curvature=0.0):
        self.N=cluster_size; self.M=n_clusters; self.mode=mode; self.kappa=curvature
    def emit(self,t):
        I=np.zeros_like(t)
        for _ in range(self.M):
            I += I_SR(self.N,t) if self.mode=="hypothesis" else I_incoh(self.N,t)
        # Krümmung: HYP ↑, ANT ↓
        gain = (1+0.5*self.kappa) if self.mode=="hypothesis" else 1/(1+2*self.kappa)
        return I*gain

    def total_photons(self, I, t): return float(np.trapz(I,t)/(E))

    def phase_reset(self, I):
        # robustere, schwächere Kopplung
        return 5e-3 * (I.max() / (I.mean() + 1e-30))
