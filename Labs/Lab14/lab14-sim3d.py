# traveling galactic salesman problem
#
# - cities in 3D unit cube [0,1) x [0,1) x [0,1)
# - cost function to minimize is total Euclidian pathlength
#
#
# by Denes Molnar for PHYS 580, Fall 2020


import numpy as np
from math import sqrt, exp
from random import random


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# COST function ("energy")
#
def calculateE(path, cities):
   N = path.shape[0]
   E = 0.
   i = path[0]
   for k in range(1, N):    # do N-1,0 separately (avoids taking modulo unnecessarily)
      j = path[k]
      dx = cities[i,0] - cities[j,0]
      dy = cities[i,1] - cities[j,1]
      dz = cities[i,2] - cities[j,2]
      E += sqrt(dx**2 + dy**2 + dz**2)
      i = j
   j = path[0]
   dx = cities[i,0] - cities[j,0]
   dy = cities[i,1] - cities[j,1]
   dz = cities[i,2] - cities[j,2]
   E += sqrt(dx**2 + dy**2 + dz**2)
   return E


# UPDATE algorithms
#

# swap 2 random sites (both picked uniformly randomly)
def update1(path):
   N = path.shape[0]
   i1 = int(random() * N)
   i2 = int(random() * (N - 1))
   if i1 == N:  i1 -= 1
   if i2 == N - 1: i2 -= 1
   if i2 >= i1: i2 += 1
   path[i1], path[i2] = path[i2], path[i1]


# reverse a random subpath
def update2(path):
   N = path.shape[0]
   i1 = int(random() * N)
   if i1 == N: i1 -= 1
   # uniformly random length 2..(N-2)  (reversing N-1 or N is just relabeling, and 1 does nothing)
   l = int(random() * (N - 3)) + 2
   if l == N - 1:  l -= 1
   i2 = i1 + l - 1
   # reverse
   for k in range(l // 2):
      j1 = (i1 + k) % N
      j2 = (i2 - k) % N
      path[j1], path[j2] = path[j2], path[j1]


# cut a random subpath and splice it elsewhere
def update3(path):
   N = path.shape[0]
   # uniformly random start 0..(N-1)
   i1 = int(random() * N)
   if i1 == N: i1 -= 1
   # uniformly random length 1..(N-2)  (length = 1 makes sense in this case)
   l = int(random() * (N - 2)) + 1
   if l == N - 1:  l -= 1
   # copy path to temporary
   path2 = np.array(path)
   # update path using temporary
   # - invariance under cyclic perms -> set beginning of spliced substring to be pos 0
   # - so segment [i1:(i1+l)] -> [0:l]
   for i in range(l):
      path[i] = path2[(i1 + i) % N]
   # uniformly random offset for remainder 1..(N-l-1)    (offs = 0 means splicing back to original spot)
   # - segment [(i1+l):i1] rotated by offset, then moved to [l:N]
   offs = int(random() * (N - l - 1)) + 1
   if offs == N - l:  offs -= 1
   for i in range(N - l):
      j = (i + offs) % (N - l)
      path[l + i] = path2[(i1 + l + j) % N]




# READ PARAMS
# 

# city location data
fname = input('city data file, empty string means random generation: ').strip()
# initial annealing temperature
T = float( input('initial annealing temperature (T): ') )

# computational params
method   = int(   input('update method [1,2,3]: ') )
Ntherm   = int(   input('update calls at each T to equilibrate: ') )
dT       = float( input('fractional reduction in T after equilibration: ') )
plotFreq = int(   input('update calls between plots: ') )


# obtain city locations
#

# cities from file
if fname != "":
   print("# reading file " + fname + "...")
   lst = []
   with open(fname, "r") as f:
      for l in f:
         [x,y,z] = l.strip().split(",")
         lst.append( [x, y, z] )
   # convert to numpy array
   N = len(lst)
   cities = np.zeros( (N, 3), dtype = float )
   for i in range(N):
      cities[i] = lst[i]

# or random generation
else:
   print("# random generation...")
   N = int( input('  number of cities (N): ') )
   cities = np.zeros( (N, 3), dtype = float )
   for i in range(N): 
      cities[i][0] = random()
      cities[i][1] = random()
      cities[i][2] = random()


# create plot window
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim( (0., 1.) )
ax.set_ylim( (0., 1.) )
ax.set_zlim( (0., 1.) )

# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


# initial path
# - a path is a permutation of cities, given in terms of city indices in order of visit
# - start with the order 0, 1, 2, .... N-1

path0 = np.zeros(N, dtype = int)
for i in range(N):
   path0[i] = i

# compute list of coordinates
rvals0 = np.zeros( (N + 1, 3), dtype = float)
for i in range(N):
   rvals0[i] = cities[path0[i]]
rvals0[N] = cities[path0[0]]

# energy
E0 = calculateE(path0, cities)



# pick update method
#

if method == 1:       update = update1
elif method == 2:     update = update2
else:                 update = update3



# interaction loop
#

path1 = np.array(path0)   # create a copy
rvals1 = np.array(rvals0)
E1 = E0

Ntotal, Nacc = 0, 0
change = True

Nupdates = 0

while True:

   # plot old and new paths
   ax.clear()
   ax.plot( rvals0[:,0], rvals0[:,1], rvals0[:,2], linewidth = linewidth, color = "green" )
   ax.plot( rvals1[:,0], rvals1[:,1], rvals1[:,2], linewidth = linewidth, color = "red" )
   #
   ax.set_xlabel("x", fontsize = fontsize)
   ax.set_ylabel("y", fontsize = fontsize)
   ax.set_zlabel("z", fontsize = fontsize)
   titleStr  = ("E=%0.5e" % E1) + (", T=%0.5e" % T) + (", dT=%0.5e" % dT) + "\n"
   titleStr += "tot=" + str(Ntotal) + ", acc=" + str(Nacc)
   ax.set_title(titleStr, fontsize = fontsize)
   plt.pause(0.001)

   # if found improvement, update old energy/path/coordinates
   if change:
      E0 = E1
      path0 = path1
      rvals0 = np.array(rvals1)


   # user input
   if Nupdates < Ntherm:    # if still thermalizing
      key = input("[" + str(Nupdates) + " done] - (q)uit, or continue: ")
      if key == "q": break

   else:                    # done thermalizing
      # adjust T
      Nupdates = 0
      T *= (1. - dT)
      #
      key = input("(q)uit, (h)alve dT, (d)ouble dT, (r)everse dT, or update: ")
      if key == "q": break
      elif key == "h":  dT *= 0.5
      elif key == "d":  dT *= 2.
      elif key == "r":  dT *= -1.

   # update
   # - path2 is our latest attempt
   # - path1 is best improvement so far
   change = False
   E1 = E0
   path1 = np.array(path0)
   for i in range(plotFreq):
      path2 = np.array(path1)
      update(path2)
      E2 = calculateE(path2, cities)
      # Metropolis update
      dE = E2 - E1
      if dE <= 0.  or  (dE > 0. and T > 0. and random() < exp(-dE / T)):
         change = True
         Nacc += 1
         E1 = E2
         path1 = path2

   # update coordinates on change
   if change:
      for i in range(N):
         rvals1[i] = cities[path1[i]]
      rvals1[N] = cities[path1[0]]

   # track efficiency
   Nupdates += plotFreq
   Ntotal += plotFreq



# print best path at end

print("# BEST PATH")
for i in range(N):
   print(i, path1[i])

#EOF
