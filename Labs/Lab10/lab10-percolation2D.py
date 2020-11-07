import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from . import percolate


# def percolate(p, grid, DBG = False):
#    L = grid.shape[0]
#    grid.fill(0)
#    labels = np.zeros(L * L, dtype = int)
#    lbl = 0
#    for i in range(L):
#       for j in range(L):
#          if np.random.rand() < p:
#             topIdx  = grid[i, j - 1] if j > 0 else 0
#             leftIdx = grid[i - 1, j] if i > 0 else 0
#             while labels[ topIdx] < 0:  topIdx = -labels[ topIdx]
#             while labels[leftIdx] < 0: leftIdx = -labels[leftIdx]
#             if topIdx == 0 and leftIdx == 0:
#                lbl += 1
#                trueIdx = lbl
#             elif topIdx == 0 and leftIdx != 0:   trueIdx = leftIdx
#             elif topIdx != 0 and leftIdx == 0:   trueIdx = topIdx
#             else:
#                if topIdx != leftIdx:
#                   idxLO = min(topIdx, leftIdx)
#                   idxHI = max(topIdx, leftIdx)
#                   trueIdx = idxLO
#                   labels[idxLO] += labels[idxHI]
#                   labels[idxHI] = -idxLO
#                else:
#                   trueIdx = leftIdx
#             grid[i,j] = trueIdx
#             labels[trueIdx] += 1
#    Noccupied = 0
#    idxMax, sMax = 0, 0
#    for idx in range(1, lbl + 1):
#       s = labels[idx]
#       if s >= 0:
#          Noccupied += s
#          if s > sMax:
#             idxMax, sMax = idx, s
#    clusterMax = np.zeros( (sMax, 2), dtype = int )
#    clustersRest = []
#    k = 0
#    for i in range(L):
#       for j in range(L):
#          if grid[i,j] != 0:
#             idx = grid[i,j]
#             while labels[idx] < 0: idx = -labels[idx]
#             if idx == idxMax:
#                clusterMax[k] = [i, j]
#                k += 1
#             else:
#                clustersRest.append([i, j])
#    clustersRest = np.asarray(clustersRest)
#    Nclusters = 0
#    sDist = {}
#    for idx in range(1, lbl + 1):
#       s = labels[idx]
#       if s >= 0:
#          Nclusters += 1
#          if sDist.get(s) is None:
#             sDist[s]  = 1
#          else:
#             sDist[s] += 1
#    sDist = sorted(sDist.items())
#    return clusterMax, clustersRest, Noccupied, Nclusters, sDist



# read parameters
#
# p        = float( input('site occupation probability (p): ') )
# L        = int( input('lattice size (L): ') )
# Nsamples = int( input('number of realizations: ') )
# clustFile = input('maximal cluster output filename: ').strip()
# distFile  = input('size distribution output filename: ').strip()


p        = 0.5  # site occupation probability
L        = 500  # lattice size
Nsamples = 5  # number of realizations
clustFile = 'boi'  # maximal cluster output filename
distFile  = 'boi_2'  # size distribution output filename


# truncate files to zero length
if clustFile != "":
   with open(clustFile, "w+"):  pass
if distFile != "":
   with open(distFile, "w+"):  pass


# prepare plot window
#

#pointarea = 20.
pointarea = 80.
pointarea *= (40. / L)**2   # scale to L = 40 

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

grid = np.zeros( (L, L), dtype = int)
for trial in range(Nsamples):

   # generate one realization
   clusterMax, clustersRest, Noccupied, Nclusters, sDist = percolate(p, grid)

   # some diagnostics 
   print(f"#DONE - trial #{trial}, Noccupied: {Noccupied}, Nclusters: {Nclusters}, sMax={sDist[-1][0]}")

   # write output to files - NOTE: it appends to existing files
   if clustFile != "":
      with open(clustFile, "a") as f:
         # include a header line
         f.write("# trial " + str(trial) + "\n")
         for (i,j) in clusterMax:
            f.write(str(i) + " " + str(j) + "\n")

   if distFile != "":
      with open(distFile, "a") as f:
         # include a header line
         f.write("# trial " + str(trial) + "\n")
         for (s,Ns) in sDist:
            f.write(str(s) + " " + str(Ns) + "\n")

   # plot sites - SKIP if you want speed
   ax0.clear()
   ax0.set_xlabel("x")
   ax0.set_ylabel("y")
   ax0.set_title("p=" + str(p) + ", L=" + str(L) + ", trial " + str(trial))
   ax0.scatter(clusterMax[:,0], clusterMax[:,1], s = pointarea, color = "red", marker = "s")
   if clustersRest != []:
      ax0.scatter(clustersRest[:,0], clustersRest[:,1], s = pointarea, color = "blue", marker = "o")

   # estimate percolation probability P(p) and susceptibility chi based on just this one sample
   P, chi = 0., 0.
   for (s, Ns) in sDist[:-1]:
      P += s * Ns
      chi += s * s * Ns
   P = 1. - P / Noccupied
   chi /= L * L          # normalize Ns -> ns

   print( "P(p)=" + str(P) + ", chi(p)=" + str(chi) )

   plt.savefig(f'plots/trial_{trial}.png', dpi=300)

