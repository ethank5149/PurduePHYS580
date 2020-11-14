import numpy as np
from numpy.random import default_rng
from itertools import product
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.integrate import trapz


class ClusterGrowth():
    def __init__(self, rows=150, cols=150, percent_fill=0.35):
        self.rows = rows
        self.cols = cols
        self.n_iterations = int(percent_fill * self.rows * self.cols)
        self.rng = default_rng()
        self.nodes = tuple(product(np.arange(self.rows), np.arange(self.cols)))
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.seed = tuple(self.rng.choice(self.nodes))
        self.cluster = set([self.seed,])
        self.grid[self.seed] = 1


    def reset(self):
        self.grid = np.zeros((self.rows, self.cols))


    def generate_grid(self):
        for (i, j) in self.cluster:
            self.grid[i, j] = 1


    def perimeter(self):
        neighbors = set([])
        for element in self.cluster:
            i, j = element  # Unpack
            neighbors.update([
                ((i - 1) % self.rows,                   j), 
                ((i + 1) % self.rows,                   j), 
                (                  i, (j + 1) % self.cols), 
                (                  i, (j - 1) % self.cols)])
        return tuple(neighbors.difference(self.cluster))


    def iterate(self):
        choice = tuple(self.rng.choice(self.perimeter()))  # Choose a random site on the perimeter
        self.cluster.add(choice)  # Add site to the cluster


    def center_grid(self):
        igrid = np.arange(self.grid.shape[0])
        jgrid = np.arange(self.grid.shape[1])
        ii, jj = np.meshgrid(igrid, jgrid)
        mass = trapz(trapz(self.grid))
        m_i = int(round(trapz(trapz(ii * self.grid, axis=0)) / mass))
        m_j = int(round(trapz(trapz(jj * self.grid, axis=1)) / mass))
        shift_i = m_i - self.grid.shape[0] // 2
        shift_j = m_j - self.grid.shape[1] // 2
        self.grid = np.roll(np.roll(self.grid, shift_i, axis=0), shift_j, axis=1)
        if self.grid[self.rows//2,self.cols//2] == 0:  # Corrects for when the cluster is evenly divided between the four corners
            self.grid = np.roll(np.roll(self.grid, self.rows//2, axis=0), self.cols//2, axis=1)
            self.center_grid()
            

    def run(self):
        for i in trange(self.n_iterations):  # For each iteration
            self.iterate()  # Iterate the growth algorithm
        self.generate_grid()  # Finally, update the grid
        self.center_grid()


def main():
    p = ClusterGrowth()
    p.run()
    fig, ax = plt.subplots(1,1, figsize=(16,16), dpi=200, constrained_layout=True)
    ax.set_title('Cluster Growth - Eden Model')
    ax.imshow(p.grid)
    plt.savefig('Output')


if __name__ == '__main__':
    main()