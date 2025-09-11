# adapters/kmc_motors.py
import numpy as np

class MotorLattice:
    def __init__(self, mt_graph):
        self.mt_graph = mt_graph
        self.pos = 0

    def step_series(self, I_t):
        Imax = max(float(I_t.max()), 1e-30)
        for It in I_t:
            p_fwd = 0.5 + 0.1*(It/Imax)     # 0.5..0.6
            p_fwd = min(max(p_fwd, 0.0), 1.0)
            self.pos += 1 if np.random.rand() < p_fwd else -1

    def net_displacement(self):
        return self.pos
