import numpy as np
from itertools import product
from tqdm.notebook import trange, tqdm


class EdenCluster():
    def __init__(self, size=500, fill=0.15):
        self.size = size
        self.ii, self.jj = np.meshgrid(range(self.size), range(self.size))
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.simulation_time = int(fill * self.size ** 2)
        self.rng = np.random.default_rng()
        self.nodes = tuple(product(np.arange(self.size), np.arange(self.size)))
        self.seed = tuple(self.rng.choice(self.nodes))
        self.cluster = set([self.seed, ])
       
    
    def perimeter(self):
        neighbors = set([])
        for (i, j) in self.cluster:
            neighbors.update([
                ((i - 1) % self.size,                   j), 
                ((i + 1) % self.size,                   j), 
                (                  i, (j + 1) % self.size), 
                (                  i, (j - 1) % self.size)])
        return tuple(neighbors.difference(self.cluster))
        
        
    def iterate(self):
        try:
            choice = tuple(self.rng.choice(self.perimeter()))  # Choose a random site on the perimeter
            self.cluster.add(choice)  # Add site to the cluster
        except ValueError:  # Catch Empty Perimeters
            pass

        
    def CoM(self):
        M = np.sum(self.grid)
        Mi = np.sum(self.ii * self.grid) / M
        Mj = np.sum(self.jj * self.grid) / M
        return int(round(Mi)), int(round(Mj))

    
    def generate_grid(self):
        for (i, j) in self.cluster:
            self.grid[i, j] = 1

            
    def center(self):
        mi, mj = self.CoM()        
        self.grid = np.roll(self.grid, (self.size // 2 - mi, self.size // 2 - mj), (1, 0))

    
    def run(self):
        
        for i in trange(self.simulation_time, desc='Growing Cluster'):
            self.iterate()
        
    def simulate(self):
        self.run()
        self.generate_grid()
        self.center()
        return self.grid