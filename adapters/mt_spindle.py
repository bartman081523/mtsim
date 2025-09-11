# adapters/mt_spindle.py
import numpy as np

class FourStateMT:
    """
    Minimaler 4-State-Port:
    S = {GROW, SHRINK, PAUSE, TRANSITION}
    Längenentwicklung in nm je Zeitschritt; Raten in 1/s.
    Parameter sind bewusst als Defaults gesetzt und sollen von dir kalibriert werden.
    """
    GROW, SHRINK, PAUSE, TRANS = 0, 1, 2, 3

    def __init__(self,
                 v_grow_nm_s=300,     # typ. 0.2–0.6 µm/s
                 v_shrink_nm_s=600,   # typ. 0.5–1.0 µm/s
                 k_cat_s=0.2,         # Katastrophe-Rate
                 k_res_s=0.05,        # Rescue-Rate
                 k_pause_s=0.05,      # in Pause wechseln
                 k_unpause_s=0.2,     # Pause verlassen
                 length_nm_init=2000  # Startlänge
                 ):
        self.vg = v_grow_nm_s
        self.vs = v_shrink_nm_s
        self.k_cat = k_cat_s
        self.k_res = k_res_s
        self.k_pause = k_pause_s
        self.k_unpause = k_unpause_s
        self.L = float(length_nm_init)
        self.state = self.GROW

    def step(self, dt_s, rng):
        # Zustandsübergänge (exponentielle Hazard-Modelle)
        if self.state == self.GROW:
            if rng.random() < 1 - np.exp(-self.k_cat*dt_s):
                self.state = self.SHRINK
            elif rng.random() < 1 - np.exp(-self.k_pause*dt_s):
                self.state = self.PAUSE
        elif self.state == self.SHRINK:
            if rng.random() < 1 - np.exp(-self.k_res*dt_s):
                self.state = self.GROW
            elif rng.random() < 1 - np.exp(-self.k_pause*dt_s):
                self.state = self.PAUSE
        elif self.state == self.PAUSE:
            if rng.random() < 1 - np.exp(-self.k_unpause*dt_s):
                # zufällig in Grow oder Shrink zurück
                self.state = self.GROW if rng.random() < 0.7 else self.SHRINK
        # Längenupdate
        if self.state == self.GROW:
            self.L += self.vg * dt_s
        elif self.state == self.SHRINK:
            self.L = max(0.0, self.L - self.vs * dt_s)

        return self.L, self.state


class MTSpindleSystem:
    """
    Bündel aus N Mikrotubuli; „Spindel-Metrik“ ~ Summe der Längen
    (optional moduliert durch ROS/Spin-Signal).
    """
    def __init__(self, n_mt=9, length_um=3.0, seed=1234):
        self.n_mt = n_mt
        self.target_len_nm = length_um*1000.0
        self.rng = np.random.default_rng(seed)
        self.mts = [FourStateMT(length_nm_init=self.target_len_nm/2) for _ in range(n_mt)]
        self.last_sum_len = self.sum_length_nm()

    def sum_length_nm(self):
        return sum(mt.L for mt in self.mts)

    def update_dynamics(self, dt_series_s, ros_modulator=1.0):
        """
        ros_modulator (>1) stabilisiert Wachstum (senkt Katastrophenrate, erhöht Rescue).
        """
        # temporär Parameter modulieren
        for mt in self.mts:
            base_kcat, base_kres = mt.k_cat, mt.k_res
            mt.k_cat = base_kcat / ros_modulator
            mt.k_res = base_kres * ros_modulator

        for dt in dt_series_s:
            for mt in self.mts:
                mt.step(dt, self.rng)

        # Parameter zurücksetzen
        for mt in self.mts:
            mt.k_cat, mt.k_res = base_kcat, base_kres

    def spindle_metric(self, ros_modulator=1.0):
        # z.B. Netto-Gewinn an Gesamtlänge relativ zum Start, mod. durch ROS
        total = self.sum_length_nm()
        dL = (total - self.last_sum_len) * ros_modulator
        self.last_sum_len = total
        return dL
