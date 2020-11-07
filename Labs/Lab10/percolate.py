import numpy as np

# Trimmed slightly for speed

def percolate(p, grid):
   L = grid.shape[0]
   grid.fill(0)
   labels = np.zeros(L * L, dtype = int)
   lbl = 0
   for i in range(L):
      for j in range(L):
         if np.random.rand() < p:
            topIdx  = grid[i, j - 1] if j > 0 else 0
            leftIdx = grid[i - 1, j] if i > 0 else 0
            while labels[ topIdx] < 0:  topIdx = -labels[ topIdx]
            while labels[leftIdx] < 0: leftIdx = -labels[leftIdx]
            if topIdx == 0 and leftIdx == 0:
               lbl += 1
               trueIdx = lbl
            elif topIdx == 0 and leftIdx != 0:   trueIdx = leftIdx
            elif topIdx != 0 and leftIdx == 0:   trueIdx = topIdx
            else:
               if topIdx != leftIdx:
                  idxLO = min(topIdx, leftIdx)
                  idxHI = max(topIdx, leftIdx)
                  trueIdx = idxLO
                  labels[idxLO] = labels[idxLO] + labels[idxHI]
                  labels[idxHI] = -idxLO
               else:
                  trueIdx = leftIdx
            grid[i,j] = trueIdx
            labels[trueIdx] = labels[trueIdx] + 1
   n_occupied = 0
   idxMax, sMax = 0, 0
   for idx in range(1, lbl + 1):
      s = labels[idx]
      if s >= 0:
         n_occupied += s
         if s > sMax:
            idxMax, sMax = idx, s
   k = 0
   for i in range(L):
      for j in range(L):
         if grid[i,j] != 0:
            idx = grid[i,j]
            while labels[idx] < 0: idx = -labels[idx]
            if idx == idxMax:
               k += 1
   sDist = {}
   for idx in range(1, lbl + 1):
      s = labels[idx]
      if s >= 0:
         if sDist.get(s) is None:
            sDist[s]  = 1
         else:
            sDist[s] += 1
   sDist = sorted(sDist.items())
   return 0, 0, n_occupied, 0, sDist


def percolate_ensemble(p0, L, n_samples):
   grid = np.zeros((L, L), dtype = int)
   P = np.zeros(n_samples)
   Chi = np.zeros(n_samples)
   
   for trial in range(n_samples):
      _, _, n_occ, _, newgrid = percolate(p0, grid)
      p, chi = 0., 0.
      for (s, Ns) in newgrid[:-1]:
         p = p + s * Ns
         chi = chi + s * s * Ns
      p = 1. - p / n_occ
      chi = chi / L * L
      P[trial] = p
      Chi[trial] = chi

   return P, Chi
