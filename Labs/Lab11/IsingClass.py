import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import trange
from os import system


class Ising:
    def __init__(
        self, 
        rows=500,         # Number of rows
        cols=500,         # Number of columns
        nsweeps=250,      # Number of Cycles
        J=1.0,            # Ferromagnetic Coupling Constant 
        h=0.0,            # Magnetic Field Strength 
        T=1.4,            # Temperature
        dpi=200,          # dpi of the figure
        figsize=(10,10),  # size of the figure
        interp='bicubic'  # Grid Interpolation 
        ):

        self.rows = rows
        self.cols = cols
        self.nsweeps = nsweeps
        self.J = J
        self.h = h
        self.T = T
        self.dpi = dpi
        self.figsize = figsize
        self.rng = default_rng()
        self.grid = 2 * self.rng.integers(low=0, high=1, endpoint=True, size=(self.rows, self.cols)) - 1  # Default to a random grid
        self.k_B = 1.0
        self.interp = interp
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        self.fig.suptitle(f"Ising Model (Metropolis-Hastings Algorithm)\n"+fr'$J={self.J:0.4f}$, $h={self.h:0.4f}$, $T={self.T:0.4f}$')


    def grid2png(self, frame):
        self.ax.imshow(self.grid, interpolation=self.interp)
        plt.savefig(f"frames/grid_{str(int(frame)).zfill(4)}.png")
        plt.cla()


    def grid2npy(self, frame):
        np.save(f"data/grid_{str(int(frame)).zfill(4)}", self.grid)


    def spin_up(self):
        self.grid = np.ones_like(self.grid)


    def spin_down(self):
        self.grid = -np.ones_like(self.grid)


    def randomize(self):
        self.grid = 2 * self.rng.integers(low=0, high=1, endpoint=True, size=self.grid.shape) - 1


    def neighbors(self, i, j):
        return  self.grid[(i - 1) % self.rows, j] + self.grid[(i + 1) % self.rows, j] + self.grid[i, (j - 1) % self.cols] + self.grid[i, (j + 1) % self.cols]


    def step(self):
        i, j = self.rng.integers(self.rows), self.rng.integers(self.cols)  # Select a random point in the grid
        dE = 2 * (self.J * self.neighbors(i, j) - self.h) * self.grid[i, j]  # Calculate change in energy of this flip
        # Accept if this flip saves energy, otherwise accept with probability $P = e^{-\frac{dE}{k_B T}}$
        if (dE < 0.0) or (self.rng.random() < np.exp(-dE / (self.k_B * self.T))):
            self.grid[i, j] *= -1


    def sweep(self, frame=0):
        for iter in range(self.grid.size):
            self.step()


    def run(self):
        for iter in trange(self.nsweeps):
                self.grid2npy(iter)
                self.sweep(iter)


    def animate(self):
        for iter in trange(self.nsweeps):
                self.grid2png(iter)
                self.sweep(iter)
        system('ffmpeg -framerate 24 -i frames/grid_%04d.png frames/grid.mp4')


if __name__ == '__main__':
    print('Using Default Settings...')
    p = Ising()
    print('Animating...')
    p.animate()
    print('Done!')
