import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class Ising:
    def __init__(self, 
        rows=25,        # Number of rows
        cols=25,        # Number of columns
        relax=500,      # Number of Equalibrium Cycles
        sample_size=100,  # Number of Cycles
        t_i=0.1,        # Initial Temperature
        t_f=3.0,        # Final Temperature
        n_t=100,       # Number of Temperature Values
        h=0.0,          # Magnetic Field Strength
        J=1.0           # Ferromagnetic Coupling Constant
        ):

        self.rows, self.cols = rows, cols
        self.relax, self.sample_size = relax, sample_size
        self.h, self.J = h, J
        self.t_i, self.t_f, self.n_t = t_i, t_f, n_t
        self.trange = np.linspace(self.t_i, self.t_f, self.n_t)
        self.rng = default_rng()
        self.k_B = 1.0
        self.grid = np.zeros((self.rows, self.cols))
        self.randomize()
        
        self.energy_series         = np.zeros(self.n_t)
        self.magnetization_series  = np.zeros(self.n_t)
        self.capacity_series       = np.zeros(self.n_t)
        self.susceptibility_series = np.zeros(self.n_t)
        self.temperature_series    = np.zeros(self.n_t)

        self.energy_sample         = np.zeros(self.sample_size)
        self.magnetization_sample  = np.zeros(self.sample_size)
        self.capacity_sample       = np.zeros(self.sample_size)
        self.susceptibility_sample = np.zeros(self.sample_size)
        self.temperature_sample    = np.zeros(self.sample_size)
        

    def randomize(self):
        self.grid = 2.0 * self.rng.integers(2, size=self.grid.shape) - 1.0


    def dE(self):
        dE = 2.0 * (self.J * self.neighbors() - self.h) * self.grid
        return dE


    def probabilities(self):
        return np.where(self.dE() < 0.0, np.ones_like(self.grid), np.exp(-self.dE() / (self.k_B * self.t)))


    def energy(self):
        return -0.5 * np.sum(self.dE())


    def magnetization(self):
        return np.sum(self.grid)


    def capacity(self):
        return 0.0


    def susceptibility(self):
        return 0.0 


    def neighbors(self):
        return np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)


    def update(self):
        self.grid = np.where(self.rng.random(size=self.grid.shape) < self.probabilities(), np.negative(self.grid), self.grid)


    def equalize(self):
        for _ in range(self.relax):
            self.update()        


    def sample(self, i=0):
        for j in range(self.sample_size):
            self.update()
            self.energy_sample[j]         = self.energy()         / self.grid.size
            self.magnetization_sample[j]  = self.magnetization()  / self.grid.size
            self.capacity_sample[j]       = self.capacity()       / self.grid.size
            self.susceptibility_sample[j] = self.susceptibility() / self.grid.size
        self.energy_series[i]         = np.mean(self.energy_sample        )
        self.magnetization_series[i]  = np.mean(self.magnetization_sample )
        self.capacity_series[i]       = np.mean(self.capacity_sample      )
        self.susceptibility_series[i] = np.mean(self.susceptibility_sample)
        self.temperature_series[i]    = self.t


    def run(self):
        for i, t in enumerate(tqdm(self.trange)):
            self.t = t
            self.equalize()
            self.sample(i)
        self.data = pd.DataFrame(
            np.vstack(( self.energy_series.flatten(), 
                        self.magnetization_series.flatten(), 
                        self.capacity_series.flatten(), 
                        self.susceptibility_series.flatten(), 
                        self.temperature_series.flatten()) ).T,
                        columns=[   'Energy', 
                                    'Magnetization', 
                                    'Specific Heat Capacity',
                                    'Magnetic Susceptibility',
                                    'Temperature'])
        self.data.to_csv('data.csv')


def main():
    p = Ising()
    p.run()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(32,32), dpi=200, constrained_layout=True)
        
    ax1.plot(p.data['Temperature'], p.data['Energy'])
    ax1.set_title('Energy vs. Temperature')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Energy')
    ax1.grid()
    
    ax2.plot(p.data['Temperature'], p.data['Magnetization'])
    ax2.set_title('Magnetization vs. Temperature')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Magnetization')
    ax2.grid()
    
    ax3.plot(p.data['Magnetization'], p.data['Energy'])
    ax3.set_title('Magnetization vs. Energy')
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Magnetization')
    ax3.grid()

    plt.savefig('IsingModel')


if __name__ == '__main__':
    main()
