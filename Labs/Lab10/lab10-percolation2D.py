# site percolation on a 2D square lattice
#
#
# by Denes Molnar for PHYS 580, Fall 2020


import numpy as np
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# computes a single realization of the percolation
#
# 2D indices: [i,j] - for horizontal,vertical => left: i-1, top: j-1
#
# grid:   - stores label of cluster the site belongs to (may be an alias)
#         - zero entry means empty site
#
# labels: - positive entries hold the site count in cluster (= cluster size)
#         - negative entries point to next alias in label chain
#         - zero entries mean unused
# 
def generate(p, grid, DBG = False):
   L = grid.shape[0]
   # zero out the grid
   grid.fill(0)
   # cluster label tracking
   labels = np.zeros(L * L, dtype = int)  # in worst case there can be ~ L^2/2 clusters...
   lbl = 0
   # fill sites with probability p (proceed in vertical scanning order)
   for i in range(L):
      for j in range(L):
         # if site will be empty, do nothing
         if random.random() > p:  continue
         # if occupied
         else:
            # check whether top & left neighbors are occupied
            # find true labels
            topIdx  = grid[i, j - 1]   if j > 0   else   0
            leftIdx = grid[i - 1, j]   if i > 0   else   0
            while labels[topIdx] < 0: topIdx = -labels[topIdx]
            while labels[leftIdx] < 0: leftIdx = -labels[leftIdx]
            # if both are empty, use a new label
            if topIdx == 0 and leftIdx == 0:
               lbl += 1
               trueIdx = lbl
            # if only one is occupied, inherit its label
            elif topIdx == 0 and leftIdx != 0:   trueIdx = leftIdx
            elif topIdx != 0 and leftIdx == 0:   trueIdx = topIdx
            # if both are occupied, inherit lower one and merge
            else:
               if topIdx != leftIdx:
                  idxLO = min(topIdx, leftIdx)
                  idxHI = max(topIdx, leftIdx)
                  trueIdx = idxLO
                  labels[idxLO] += labels[idxHI]   # merge counts from alias
                  labels[idxHI] = -idxLO           # mark alias
               else:
                  trueIdx = leftIdx
            # update grid and count
            grid[i,j] = trueIdx
            labels[trueIdx] += 1

   # print grid (if we are debugging)
   if DBG:
      for j in range(L):
         line = "".join( [ chr(grid[i,j] + 64) if grid[i,j] != 0  else "." for i in range(L) ] )
         print(line)

   # find maximal (largest) cluster, count occupied sites
   # - if multiple clusters have the same maximal size, pick first one found
   Noccupied = 0
   idxMax, sMax = 0, 0
   for idx in range(1, lbl + 1):
      s = labels[idx]
      if s < 0: continue # skip aliases
      if DBG:  print(idx, chr(idx + 64), s)
      Noccupied += s
      if s > sMax:
         idxMax, sMax = idx, s

   if DBG: print("idxMax:", idxMax, "sMax:", sMax)

   # create arrays of coordinates for the maximal cluster, and the rest
   clusterMax = np.zeros( (sMax, 2), dtype = int )
   clustersRest = []
   k = 0
   for i in range(L):
      for j in range(L):
         if grid[i,j] == 0: continue    # skip empty sites
         idx = grid[i,j]
         while labels[idx] < 0: idx = -labels[idx]    # find true label
         if idx == idxMax:
            clusterMax[k] = [i, j]
            k += 1
         else:
            clustersRest.append([i, j])
   clustersRest = np.array(clustersRest)   # convert to np array

   # collect cluster distribution and number of clusters
   # - use dictionary for faster O(ln n) duplicate identification
   Nclusters = 0
   sDist = {}
   for idx in range(1, lbl + 1):
      s = labels[idx]
      if s < 0: continue   # skip aliases
      Nclusters += 1
      if sDist.get(s) == None:  sDist[s] = 1
      else:                     sDist[s] += 1
   sDist = sorted(sDist.items())  # convert to sorted list of elements (s, Ns)

   # some crosschecks
   #if True or DBG:
   if DBG:
      assert sum( [ Ns      for (s,Ns) in sDist ]) == Nclusters
      assert sum( [ s * Ns  for (s,Ns) in sDist ]) == Noccupied

   # return 1) sites in maximal cluster, 2) sites in rest of clusters, 3) number of occupied sites
   #        4) number of clusters, and 5) cluster size distribution (includes maximal cluster)
   return clusterMax, clustersRest, Noccupied, Nclusters, sDist



# read parameters
#
p        = float( input('site occupation probability (p): ') )
L        = int( input('lattice size (L): ') )
Nsamples = int( input('number of realizations: ') )
clustFile = input('maximal cluster output filename: ').strip()
distFile  = input('size distribution output filename: ').strip()


# truncate files to zero length
if clustFile != "":
   with open(clustFile, "w+"):  pass
if distFile != "":
   with open(distFile, "w+"):  pass


# prepare plot window
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
#fontsize = 15    # fontsize, for 1920x1080
#linewidth = 1.5
linewidth = 3
#pointarea = 20.
pointarea = 80.

pointarea *= (40. / L)**2   # scale to L = 40 

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

ax0.tick_params(labelsize = fontsize)


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


# loop over percolations
#

grid = np.zeros( (L, L), dtype = int)
for trial in range(Nsamples):

   # generate one realization
   clusterMax, clustersRest, Noccupied, Nclusters, sDist = generate(p, grid)

   # some diagnostics 
   print( "#DONE - trial #" + str(trial) + ", Noccupied: " + str(Noccupied) + ", Nclusters: " + str(Nclusters)
          + ", sMax=" + str(sDist[-1][0]) )

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
   ax0.set_xlabel("x", fontsize = fontsize)
   ax0.set_ylabel("y", fontsize = fontsize)
   ax0.set_title("p=" + str(p) + ", L=" + str(L) + ", trial " + str(trial), fontsize = fontsize)
   ax0.scatter(clusterMax[:,0], clusterMax[:,1], s = pointarea, color = "red", marker = "s")
   if clustersRest != []:
      ax0.scatter(clustersRest[:,0], clustersRest[:,1], s = pointarea, color = "blue", marker = "o")
   plt.pause(0.0001)  # update plot

   # estimate percolation probability P(p) and susceptibility chi based on just this one sample
   P, chi = 0., 0.
   for (s, Ns) in sDist[:-1]:
      P += s * Ns
      chi += s * s * Ns
   P = 1. - P / Noccupied
   chi /= L * L          # normalize Ns -> ns

   print( "P(p)=" + str(P) + ", chi(p)=" + str(chi) )

   # pause for ENTER
   input("[ENTER]")


# wait for user input before closing plot window
while input("Finish [q]") != "q":   pass


#EOF
