import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm


class IsingModel(object):
    def __init__(self, ncols=10, nrows=10, J=1., kB=1., Trange=(2., 2.27), nT=100, Hrange=(-10.0, 10.0), nH=100, nR=250, nS=1000):
        self.nrows, self.ncols = nrows, ncols
        self.J, self.kB = J, kB
        self.Ti, self.Tf, self.nT = *Trange, nT
        self.Hi, self.Hf, self.nH = *Hrange, nH
        self.nR, self.nS = nR, nS
        self.Trange, self.Hrange = np.linspace(self.Ti, self.Tf, self.nT), np.linspace(self.Hi, self.Hf, self.nH)
        self.scale = 1. / (self.nrows * self.ncols)
        self.rng = np.random.default_rng()
        self.at = self.rng.choice(np.array([1, -1]), size=(self.nrows, self.ncols))
        self.Esample = np.zeros(self.nS)
        self.Msample = np.zeros(self.nS)
        self.Hs = np.zeros(self.nH * self.nT)
        self.Ts = np.zeros(self.nH * self.nT)
        self.Es = np.zeros(self.nH * self.nT)
        self.Ms = np.zeros(self.nH * self.nT)
        self.Cs = np.zeros(self.nH * self.nT)
        self.Xs = np.zeros(self.nH * self.nT)
        self.show_H_progress = True
        self.show_T_progress = True
        self.show_S_progress = True

    def metric_at(self, i, j):
        return self.at[ (i + 1)               % self.nrows,   j                                ] + \
               self.at[((i - 1) + self.nrows) % self.nrows,   j                                ] + \
               self.at[  i                                ,  (j + 1)               % self.ncols] + \
               self.at[  i                                , ((j - 1) + self.ncols) % self.ncols]

    def get_E(self):
        return self.scale * np.mean(self.Esample)
        
    def get_M(self):
        return self.scale * np.mean(self.Msample)
        
    def get_C(self, T):
        return self.scale * np.var(self.Esample) * T ** (-2.)
    
    def get_X(self, T):
        return self.scale * np.var(self.Msample) * T ** (-1.)

    def metric(self):
        _ = np.zeros_like(self.at)
        for i in range(self.nrows):
            for j in range(self.ncols):
                _[i, j] = self.metric_at(i, j)
        return _

    def update(self, H, T):
        for i in range(self.nrows):
            for j in range(self.ncols):
                dE = 2. * (self.J * self.metric_at(i, j) - H) * self.at[i, j]
                prob = 1. if dE < 0.0 else np.exp(-dE / (self.kB * T))
                if self.rng.random() < prob:
                    self.at[i, j] = -self.at[i, j]    

    def relax(self, H, T):
        for _ in range(self.nR):#, desc='Relaxing The System', leave=False):
            self.update(H, T)

    def measure_at(self, H, T):
        for _ in range(self.nS):  # trange(self.nS, desc='Collecting Sample', leave=False):
            self.update(H, T)
            dE = 2. * (self.J * self.metric() - H) * self.at
            self.Esample[_] = -.5 * np.sum(dE)
            self.Msample[_] = np.sum(self.at)
        return H, T, self.get_E(), self.get_M(), self.get_C(T), self.get_X(T)

    def reset(self):
        self.at = self.rng.choice(np.array([1, -1]), size=(self.nrows, self.ncols))
        
    def simulate(self):
        for i_h, H in enumerate(tqdm(self.Hrange, desc='Sweeping Magnetic Fields', leave=False)):
            for i_t, T in enumerate(tqdm(self.Trange, desc='Sweeping Temperatures', leave=False)):
                line = i_h * self.nT + i_t                
                self.reset()
                self.relax(H, T)
                self.Hs[line], self.Ts[line], self.Es[line], self.Ms[line], self.Cs[line], self.Xs[line] = self.measure_at(H, T)
        return pd.DataFrame(np.vstack((self.Hs, self.Ts, self.Es, self.Cs, self.Ms, self.Xs)).T, 
                            columns=['MagneticField','Temperature', 'Energy', 'SpecificHeatCapacity', 'Magnetization', 'MagneticSusceptibility'])