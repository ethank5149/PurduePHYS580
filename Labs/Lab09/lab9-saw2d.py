# Self-avoiding random walk (SAW) on a 2 dimensional square lattice
#
# by Denes Molnar for PHYS 580, Fall 2020


import random
from math import sqrt, pi as PI
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# generate one SAW, starting from origin
#
# - return 2D array x[i,k], where x[i] is the 2D coordinate after i steps,
#   and length of walk 'n' (so x[n] is the last written datapoint)
# - array returned may be shorter than Nsteps if walker did not 'survive'
# - tracks occupied sites in 'lattice', origin of lattice is at (Nsteps, Nsteps)
#
def generateWalk(Nsteps, lattice):

   # four principle directions of walk (2D)
   # - opposite directions are together in groups of 2
   #   => inverse of direction j is (j ^ 1), where ^ is the xor operation
   dirs = ( (1, 0), (-1, 0), (0, 1), (0, -1) )

   # initialize storage for walk
   x = np.zeros( (Nsteps + 1, 2), dtype = int )

   # start from origin
   x[0] = np.array([0, 0])
   lattice[Nsteps, Nsteps] = 1

   # first step in any direction
   newdir = random.randint(0, 3)    #includes 3(!)
   x[1] = dirs[newdir]
   lattice[x[1][0] + Nsteps, x[1][1] + Nsteps] = 1
   revdir = newdir ^ 1       # reverse of move we picked

   # follow with steps 2, 3, ... Nsteps
   for n in range(2, Nsteps + 1):
      # generate a direction that is DIFFERENT from revdir (no going back)
      newdir = random.randint(0, 2)
      if newdir >= revdir: newdir += 1
      # store new position, terminate if occupied
      x[n] = x[n - 1] + dirs[newdir]
      if lattice[ x[n,0] + Nsteps, x[n,1] + Nsteps] != 0:
         n -= 1
         break
      # mark site occupied, update reverse direction
      lattice[ x[n,0] + Nsteps, x[n,1] + Nsteps] = 1
      revdir = newdir ^ 1

   # restore lattice to empty
   for i in range(n + 1):
      lattice[ x[i,0] + Nsteps, x[i,1] + Nsteps] = 0

   # return walk - last element written was [n]
   return x[:(n + 1)], n



# read physical parameters
#
Nsteps = int( input('maximum number of steps: ') )
Nwalks = int( input('number of walks: ') )
cutoff = float( input('min fraction of surviving walks below which to cut off: ') )

# default Python RNG (Mersenne Twister) is good enough
#

# generate and plot random walks, one by one,
# then do <r^2> analysis
#

#fontsize = 30   # fontsize, need to be tuned to screen resolution
fontsize = 15    # fontsize, for 1920x1080
linewidth = 1.5
#linewidth = 3
pointarea = 20.
#pointarea = 50.

# one 2D figure
fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

ax0.set_xlabel("x")
ax0.set_ylabel("y")

# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


# lattice for tracking occupancy (0 empty, 1 occupied,  origin at (Nsteps, Nsteps)
lattice = np.zeros((2 * Nsteps + 1, 2 * Nsteps + 1) )
# survivor counts
survivors = np.zeros(Nsteps + 1)
# storage for <r^2> statistics
r2avg = np.zeros(Nsteps + 1)

# generate random walks, track number of survivors
#
for j in range(Nwalks):
   x, n = generateWalk(Nsteps, lattice)
   #track survivor counts
   for i in range(n + 1):
      survivors[i] += 1
   # plot the path - SKIP IT, if speed matters
   ax0.clear()
   ax0.plot(x[:,0], x[:,1], linewidth=linewidth, color = "blue")
   ax0.set_title("Self-avoiding random walk #" + str(j) )
   plt.pause(0.0001)  # redraw
   input("[ENTER]")

   # collect r^2 statistics vs number of steps
   for i in range(n + 1):
      r2avg[i] += x[i][0]**2 + x[i][1]**2


# convert sum(r^2) data to <r^2>
#

# find cutoff for walk length
for ncutoff in range(Nsteps + 1):
   if survivors[ncutoff] < cutoff * Nwalks:
      break
# truncate sum(r^2) data and scale to <r^2>
r2avg = r2avg[:ncutoff]
for i in range(ncutoff):
   r2avg[i] /= survivors[i]

#print(survivors[:ncutoff])


# plot <r^2> vs n in log-log plot
#

tvals = [ i for i in range(ncutoff) ]  # t values (steps) for horizontal axis

logtVals  = np.log(tvals[1:]) / np.log(10)    # log_10 of n
logr2Vals = np.log(r2avg[1:]) / np.log(10)    # log_10 or <r^2>

ax0.clear()
ax0.set_xlabel('log10( t )')
ax0.set_ylabel('log10( <r^2> )')
ax0.scatter(logtVals, logr2Vals, s = pointarea, color = "red", marker = "x")

# fit an 'a*x+b' line to <r^2> vs n, then print & plot the fit
#
poly = np.polyfit(logtVals, logr2Vals, 1)
(a, b) = poly

# error on 'a' under normality assumption
# (see https://en.wikipedia.org/wiki/Simple_linear_regression)
res = sum( [ (a*x + b - y)**2  for  x,y in zip(logtVals, logr2Vals) ] )
avx = sum( logtVals ) / Nsteps
varx = sum( [ (x - avx)**2  for x in logtVals ] )
da = sqrt( res / (varx * (Nsteps - 2)) )

print( "\n")
print( "polynomial fit to log10(<r^2>) vs log10(t):" )
print( "" )
print( "   log10<r^2> = " + str(a) + " * log10(t) + " + str(b) )
print( "" )
print( "   da=", da )
print("")


# plot the fit
ax0.plot(logtVals, a * logtVals + b, linewidth = linewidth, color = "blue")



# demand user input before closing plot window
while input("Finish [q]") != "q":
   continue


#EOF
