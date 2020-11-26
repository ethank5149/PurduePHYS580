import numpy as np
import pandas as pd
from itertools import product
from tqdm.notebook import trange, tqdm


class ClusterGrowth():
    def __init__(self, rows=250, cols=250, fill=0.15):
        self.rows = rows
        self.cols = cols
        
        ## See note under function `CoM`
        if self.rows != self.cols:
            print('Warning: Only square grids are supported currently, using larger length for both sides...')
            if self.rows > self.cols:
                self.cols = self.rows
            else:  # self.rows < self.cols
                self.rows = self.cols
                
        self.ii, self.jj = np.meshgrid(range(self.rows), range(self.cols))
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.maxM = self.grid.size
        self.maxR = np.sqrt(self.maxM / 2.)
        self.Rs, self.Ms = np.zeros_like(self.grid), np.zeros_like(self.grid)

        self.simulation_time = int(fill * self.rows * self.cols)
        self.rng = np.random.default_rng()
        self.nodes = tuple(product(np.arange(self.rows), np.arange(self.cols)))
        self.seed = tuple(self.rng.choice(self.nodes))
        self.cluster = set([self.seed,])
                
    def generate_grid(self):
        for (i, j) in self.cluster:
            self.grid[i, j] = 1

    def center(self):
        mi, mj = self.CoM()        
        self.grid = np.roll(self.grid, (self.rows // 2 - mi, self.cols // 2 - mj), (1, 0))
        
    def perimeter(self):
        neighbors = set([])
        for (i, j) in self.cluster:
            neighbors.update([
                ((i - 1) % self.rows,                   j), 
                ((i + 1) % self.rows,                   j), 
                (                  i, (j + 1) % self.cols), 
                (                  i, (j - 1) % self.cols)])
        return tuple(neighbors.difference(self.cluster))
        
    def pseudo_radius(self):
        return np.sqrt(np.sum(self.grid) / np.pi) / self.maxR
    
    def iterate(self):
        try:
            choice = tuple(self.rng.choice(self.perimeter()))  # Choose a random site on the perimeter
            self.cluster.add(choice)  # Add site to the cluster
        except ValueError:
            pass

    def CoM(self):  ## FIXME: When rows != cols, matrix ops break... :(
        M = np.sum(self.grid)
        Mi = np.sum(self.ii * self.grid) / M
        Mj = np.sum(self.jj * self.grid) / M
        return int(round(Mi)), int(round(Mj))

    def radial(self):
        ci, cj = self.CoM()
        R = np.sqrt((self.ii - ci) ** 2. + (self.jj - cj) ** 2.)
        M = np.zeros_like(self.grid)
        
        for (i, j) in tqdm(self.nodes, desc='Analyzing Cluster'):
            M[i, j] = np.sum(np.where(R <= R[i, j], self.grid, np.zeros_like(self.grid)))
        return R.flatten() / self.maxR, M.flatten() / self.maxM

    def run(self):        
        for i in trange(self.simulation_time, desc='Growing Cluster'):
            self.iterate()
        
    def simulate(self):
        self.run()
        self.generate_grid()
        self.center()
        R, M = self.radial()
        return pd.DataFrame(np.vstack((R, M)).T, columns=['Radius','Mass'])