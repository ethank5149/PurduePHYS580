# Ising model on a square lattice (2D)
#
#
# by Denes Molnar for PHYS 580, Fall 2020



import numpy as np
import random
from math import exp

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# one Monte Carlo sweep over the grid
# - grid of spins is updated in place
def updateMC(spins, params, diagnostics = False):
   (N, T, H) = params

   # do one sweep over grid
   dmTot, dETot = 0, 0.
   flipped = [[],[]]   # collect new reds and new blues
   for i in range(N):
      for j in range(N):
         # compute energy change if we flip spin at [i,j]
         sum = spins[(i-1) % N,j] + spins[(i+1) % N,j] + spins[i,(j-1) % N] + spins[i,(j+1) % N]
         dE = 2 * spins[i,j] * (sum + H)
         # always flip if it lowers the energy
         # otherwise, flip with Metropolis probability
         if dE < 0. or exp(-dE / T) > random.random():
            spins[i,j] = -spins[i,j]
            dmTot += 2 * spins[i,j]
            dETot += dE
            if diagnostics:
               if spins[i,j] == 1: flipped[0].append( (i, j) )
               else:               flipped[1].append( (i, j) )
   # convert lists to numpy arrays
   for i in range(2):
      if flipped[i] != []: flipped[i] = np.array(flipped[i])
   # return 1) list of spins flipped (if diagnostics was requested)
   #        2) change in magnetization, 3) change in energy
   return flipped, dmTot / N**2, dETot / N**2


# plot spin configuration for entire grid
def plotGrid(ax, spins):
   blues, reds = [], []
   for i in range(N):
      for j in range(N):
         if spins[i,j] == -1:
            blues.append([i,j])
         else:
            reds.append([i,j])

   blues = np.array(blues) # down spins
   reds = np.array(reds)   # up spins

   plotRedsBlues(ax0, reds, blues)


def plotRedsBlues(ax0, reds, blues):
   if reds != []:
      ax0.scatter(reds[:,0], reds[:,1], s = pointarea, color = "red", marker = "o")
   if blues != []:
      ax0.scatter(blues[:,0], blues[:,1], s = pointarea, color = "blue", marker = "o")




# read parameters
#
N   = 50  # Lattice size N (for NxN lattice)
T   = 1.5  # Temperature [J/k_B]
H   = 0.0  # Magnetic field [J/mu]
MCS = 100  # Monte Carlo steps per spin
fill = 'random'
icType = 2  # Initial spin config (1: all up, -1: all down, 2: random)
fname  = 'output'  # output file for magnetization/energy time series
params = (N, T, H)


# initial conditions
#

if fill is 'random':
   spins = 2 * np.random.randint((N,N), dtype = int ) - 1
elif fill is 'up':
   spins = np.ones((N,N), dtype=int)
elif fill is 'down':
   spins = -np.ones((N,N), dtype=int)

# compute initial energy and magnetization
E = 0.
for i in range(N):
   for j in range(N):
      # nbs = spins[(i-1) % N,j] + spins[(i+1) % N, j] + spins[i,(j-1) % N] + spins[i,(j+1) % N]
      E -= 0.5 * spins[i,j] * spins[(i-1) % N,j] + spins[(i+1) % N, j] + spins[i,(j-1) % N] + spins[i,(j+1) % N] - H * spins[i,j]
m = np.sum(spins)
intrinsic_m = m / spins.size
intrinsic_E = E / spins.size


# open output file (truncate)
# 
if fname != "":
   f = open(fname, "w+")
else:
   f = None


# display initial spin configuration
#

pointarea = 80. * (20. / N)**2   # scale to N = 20 

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray


plotGrid(ax0, spins)
plt.show()





# do Monte Carlo updates
# 
diagFreq = 1 # frequency of diagnostic output to stdout
#diagFreq = 10

Nscatter = 0   # track number of points in scatter plot to prevent eventual slowdown
iter = 0
while True:
   if f != None:
      f.write(str(iter) + " " + str(m) + " " + str(E) + "\n")
      f.flush()
   # termination condition
   if iter == MCS: break
   # do one update
   iter += 1
   flipped, dE, dm = updateMC(spins, params, False)  # update without diagnostics
   m += dm
   E += dE
   # update plot
   if flipped[0] != [] or flipped[1] != []:
      # if too many repeated points already, then clear and plot entire grid
      if Nscatter > 2 * N * N:
         ax0.clear()
         plotGrid(ax0, spins)
         Nscatter = 0
      # otherwise, only replot sites that changed
      else:
         plotRedsBlues(ax0, flipped[0], flipped[1])
         Nscatter += len(flipped[0]) + len(flipped[1]) # track number of points replotted

      titleStr = "MCS=" + str(iter) + ", m=" + str(round(m, 4)) + ", E=" + str(round(E, 4)) + "\n"
      ax0.set_title(titleStr)
      plt.pause(0.001)
      input('[ENTER]')

# close file
if f != None:  f.close()

