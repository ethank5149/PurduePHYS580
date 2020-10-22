# Random walk on a 2 dimensional square lattice
#
# by Denes Molnar for PHYS 580, Fall 2020


import random
from math import sqrt, pi as PI
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# generate ONE random walk, starting from origin
# - return 2D array x[i,k], where x[i] is the 2D coordinate after i steps
#
def generateWalk(Nsteps):

   # four principle directions of walk (2D)
   dirs = ( (1, 0), (-1, 0), (0, 1), (0, -1) )

   # initialize storage for walk
   x = np.zeros( (Nsteps + 1, 2) )

   # starting from origin, take Nsteps steps
   x[0] = [0, 0]
   for i in range(Nsteps):
      r = random.random()
      if   r <= 0.25:  x[i + 1] = x[i] + dirs[0]
      elif r <= 0.5:   x[i + 1] = x[i] + dirs[1]
      elif r <= 0.75:  x[i + 1] = x[i] + dirs[2]
      else:            x[i + 1] = x[i] + dirs[3]

   # return result
   return x



# read physical parameters
#
Nsteps = int( input('maximum number of steps: ') )
Nwalks = int( input('number of walks: ') )

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


# storage for <r^2> statistics
r2avg = np.zeros(Nsteps + 1)

# generate random walks, one by one
for j in range(Nwalks):
   x = generateWalk(Nsteps)
   # plot the path - SKIP IT, if speed matters
   ax0.clear()
   ax0.plot(x[:,0], x[:,1], linewidth=linewidth, color = "blue")
   ax0.set_title("Random walk #" + str(j) )
   plt.pause(0.0001)  # redraw
   input("[ENTER]")

   # collect r^2 statistics vs number of steps
   for i in range(Nsteps + 1):
      r2avg[i] += x[i][0]**2 + x[i][1]**2

# convert sum(r^2) data to <r^2>
for i in range(Nsteps + 1):
   r2avg[i] /= Nwalks

# plot <r^2> vs n in log-log plot
#

tvals = [ i for i in range(Nsteps + 1) ]  # t values (steps) for horizontal axis

logtVals  = np.log(tvals[1:]) / np.log(10)    # take log_10 of n
logr2Vals = np.log(r2avg[1:]) / np.log(10)    # take log_10 or <r^2>

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
