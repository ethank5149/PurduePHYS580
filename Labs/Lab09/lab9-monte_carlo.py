# Monte Carlo integration - evaluates f(x) at randomly sampled x
#
# by Denes Molnar for PHYS 580, Fall 2020
#

import random
from math import sqrt, pi as PI
import numpy as np


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# function f(x) to integrate
def f(x): return sqrt(4. - x * x)


# read simulation parameters
n = int( input('Number of random x values per trial: ') )
Ntrials = int( input('Number of independent trials: ') )

x1, x2 = 0., 2.   # integration range

# initialize RNG
# - by default, Python uses the Mersenne Twiste, which is good enough for us
#   => nothing to do



# plots 
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

ax0.set_xlim((x1, x2))
#ax0.set_ylim((0, 1))
ax0.set_xlabel("x")
ax0.set_ylabel("f")


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


# plot the integrand vs x
#
n1 = 1000  # number of points to use
dx = (x2 - x1) / n1
xvals = np.array( [ i * dx   for i in range(n1 + 1) ] )
yvals = np.array( [ f(x)     for x in xvals ] )

ax0.plot(xvals, yvals, linewidth = linewidth, color = "blue", label = "f(x)")
plt.pause(0.01) # show curve


# do the Monte Carlo integration
# - cycle colors in visualization
#

colors = ["red", "orange", "black", "green", "cyan", "yellow"]
Ncolors = len(colors)
xx = np.zeros(n)
yy = np.zeros(n)

sum, sum2 = 0., 0.    # sum and sum-squared for integration results obtained in trials
#displayFreq = 1000
displayFreq = 50
for t in range(Ntrials):
   ax0.set_title("Monte Carlo Integration, trial " + str(t))
   # collect sum(f), do the integral
   tot = 0.
   for i in range(n):
      x = x1 + random.random() * (x2 - x1)  # uniform on [x1,x2]
      fx = f(x)
      xx[i], yy[i] = x, fx   # store for plotting
      tot += fx
   result = tot * (x2 - x1) / n
   # show set of points used in every displayFreq-th trial
   #
   # - if things run too slowly, DISABLE this by setting a high displayFreq 
   #
   if (t % displayFreq) == 0:
      colIdx = (t // displayFreq) % Ncolors   # cycle through colors
      ax0.scatter(xx, yy, s = pointarea, color = colors[colIdx])
      plt.pause(0.1)
   # update sums for statistical analysis
   sum += result
   sum2 += result**2

# print average and standard deviation over trials
#

mean = sum / Ntrials
var = sum2 / Ntrials - mean**2
stddev = sqrt(var)
err = stddev / sqrt(Ntrials)

print("n=%d, Ntrials=%d" % (n, Ntrials) )
print("mean=%0.5f, stddev=%0.5f, err=%0.5f" % (mean, stddev, err) )
print("exact=", PI)


# keep plots open until user interaction
input("[ENTER]")


#EOF
