# file: modules/spin_ros.py
import numpy as np
gamma_e=1.76e11

class SpinROS:
    def __init__(self, curvature=0.0, T2_base=1e-9):
        self.T2 = T2_base*(1+curvature)  # Steelman-Hypothese
    def modulator(self, B=1e-6, t=1e-9):
        return 0.5*(1+np.cos(gamma_e*B*t)*np.exp(-t/self.T2))
