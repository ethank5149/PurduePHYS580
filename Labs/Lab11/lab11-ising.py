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
N   = int(   input('Lattice size N (for NxN lattice): ') )
T   = float( input('Temperature [J/k_B]: ') )
H   = float( input('Magnetic field [J/mu]: ') )
MCS = int(   input('Monte Carlo steps per spin: ') )
icType = int( input('Initial spin config (1: all up, -1: all down, 2: random): ') )
fname  = input('output file for magnetization/energy time series: ').strip()

params = (N, T, H)

#diagnostics
print( "N=%d, T=%0.5f [J/k_B], H=%0.5f [J/mu]" % (N, T, H) )


# initial conditions
#

spins = np.zeros( (N,N), dtype = int )

# set spin directions
if   icType ==  1:   spins.fill( 1)
elif icType == -1:   spins.fill(-1)
else:
   for i in range(N):
      for j in range(N):
         if random.random() <= 0.5:  spins[i,j] =  1
         else:                       spins[i,j] = -1

# compute initial energy and magnetization
m, E = 0., 0.
for i in range(N):
   for j in range(N):
      m += spins[i,j]
      # nearest neighbor sum, periodic boundary conditions
      sum = spins[(i-1) % N,j] + spins[(i+1) % N, j] + spins[i,(j-1) % N] + spins[i,(j+1) % N]
      E = E - 0.5 * spins[i,j] * sum - H * spins[i,j]
m /= N**2  # normalize per site
E /= N**2


# open output file (truncate)
# 
if fname != "":
   f = open(fname, "w+")
else:
   f = None


# display initial spin configuration
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 3
pointarea = 80. * (20. / N)**2   # scale to N = 20 

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

ax0.tick_params(labelsize = fontsize)

plotGrid(ax0, spins)

# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


plt.pause(0.001)
input('[ENTER]')



# do Monte Carlo updates
# 
diagFreq = 1 # frequency of diagnostic output to stdout
#diagFreq = 10

Nscatter = 0   # track number of points in scatter plot to prevent eventual slowdown
iter = 0
while True:
   # diagnostic output and output to file (if requested)
   if (iter % diagFreq) == 0:
      print( " MCS=%d: M=%0.5f, E=%0.5f\n" % (iter, m, E) )
   if f != None:
      f.write(str(iter) + " " + str(m) + " " + str(E) + "\n")
      f.flush()
   # termination condition
   if iter == MCS: break
   # do one update
   iter += 1
   #flipped, dE, dm = updateMC(spins, params, False)  # update without diagnostics
   flipped, dm, dE = updateMC(spins, params, True)   # update with diagnostics
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


# wait for user input
while input("Finish [q]") != "q":  pass

#EOF
