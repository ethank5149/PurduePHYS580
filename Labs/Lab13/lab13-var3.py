# ground state solution of time-independent Schroedinger eqn in 1D
# via a stochastic variational method
#
# - Lennard-Jones potential in 1D on x = [0, 5 sigma L]
# - usual L=1, hbar^2/(m L^2)=1 units
# - without loss of generality, assume that psi is real
#
# NOTE: break parity with Matlab version and use trapezoid integration
# (more sensible than rectangles)
#
#
# by Denes Molnar for PHYS 580, Fall 2020
#


import numpy as np
from math import sqrt, exp
from random import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Lennard-Jones potential
def calculateV(x, epsilon, sigma):
   invr6 = (sigma / x)**6
   return 4. * epsilon * (invr6 * invr6 - invr6)


# wave function is represented by values at equidistant points x_i,
# i.e., psi_i = psi(x_i), where i = 0...N

# normalize wave function, in place
# - trapezoid integration
# - FIXME: could exploit that for our use case psi = 0 at endpoints
def normalize(psi, xvals):
   dx = xvals[1] - xvals[0]
   N = psi.shape[0] - 1
   #
   tot = 0.5 * psi[0]**2
   for i in range(1, N):
      tot += psi[i]**2
   tot += 0.5 * psi[N]**2
   norm = sqrt(dx * tot)
   psi *= 1. / norm


# compute energy as ratio of two integrals E = <psi|H|psi> / <psi|psi>
#
# <psi|H|psi> = int[ dx  psi (-0.5 laplace psi) + V |psi|^2 ]
#
# - approximates: (laplace psi)_i = (psi_i+1 + psi_i-1 - 2 psi_i) / dx^2
# - trapezoid integration
#
# psi need not be normalized
#
def computeE(psi, xvals, eps, sigma):
   dx = xvals[1] - xvals[0]
   N = psi.shape[0] - 1
   # left endpoint
   tot = 0.5 * psi[0]**2
   V0 = calculateV(xvals[0], eps, sigma)
   Etot = 0.5 * (-0.5 * psi[0] * (psi[1] - 2. * psi[0]) / dx**2 + V0 * psi[0]**2)      # psi[-1] = 0 assumed
   # middle range
   for i in range(1, N):
      tot += psi[i] ** 2
      laplace = (psi[i + 1] + psi[i - 1] - 2. * psi[i]) / dx**2
      V = calculateV(xvals[i], eps, sigma)
      Etot += -0.5 * psi[i] * laplace + V * psi[i]**2
   # right endpoint
   tot += 0.5 * psi[N]**2
   VN = calculateV(xvals[N], eps, sigma)
   Etot += 0.5 * (-0.5 * psi[N] * (psi[N-1] - 2. * psi[N]) / dx**2 + VN * psi[N]**2)   # psi[N+1] = 0 assumed
   # E = <psi|H|psi> / <psi|psi>
   return Etot / tot   # dx factors cancel out


# Monte Carlo update
# - updates psi in place
# - returns final energy, number of accepted update attempts
#
def update(Nattempts, Eold, psi, xvals, params):
   N = psi.shape[0] - 1
   (eps, sigma, dpsiMax, T) = params
   # storage for undoing changes
   psiOld = np.zeros(N + 1)
   # update loop
   E = Eold
   Naccepted = 0
   for i in range(Nattempts):
      # choose random segment [n,m]
      n = int(random() * (N - 2)) + 1      # n = 1...(N-1)
      m = n + int(random() * (N - n))      # m = n...(N-1) 
      # choose a dpsi value
      dpsi = 2. * (random() - 0.5) * dpsiMax
      # modify psi in segment
      for j in range(n, m + 1):
         psiOld[j] = psi[j]
         psi[j] += dpsi
      # Metropolis step
      Enew = computeE(psi, xvals, eps, sigma)
      dE = Enew - Eold
      if dE <= 0. or (T > 0. and random() <  exp(-dE / T)): 
         normalize(psi, xvals)    # normalize on accept
         E = Enew
         Naccepted += 1
      else:
         for j in range(n, m + 1):   psi[j] = psiOld[j]    # revert on reject
   #
   return E, Naccepted



# read parameters
#

epsilon  = float( input('L-J energy scale [hbar^2/(mL^2)], epsilon= ') )
sigma    = float( input('L-J length scale [in units of L], sigma= ') )

# computational params
dx        = float( input('grid spacing [units of L], dx= ') )
dpsiRel   = float( input('max dpsi relative to initial wave function height: ') )
Nattempts = int(   input('number of iterations between plots: ') )
T         = float( input('initial annealing temperature: ') )
fname     = input('output file name for final psi(x): ').strip()


# hardcoded parameters
#
# integration range [xleft,xright] 
# NOTE: minimum of potential is at x = 2**(1/6) * sigma =~ 1.1 sigma
xleft = 0.5 * sigma
xright = 5. * sigma
#
# x1, x2 for initial condition
x1, x2 = 1.1 * sigma, 2.6 * sigma
# horizontal plot range [0, xmax]
xmax = 5. * sigma
# vertical plotting range [-ymax, ymax]
ymax = 2.


# compute x_i values
#

N = int((xright - xleft) / dx) + 1
xvals = np.zeros(N + 1)

for i in range(N + 1):
   xvals[i] = xright - (N - i) * dx


# initial trial wave function: constant on [x1, x2], 0 outside
#
# psi[i] stores values psi(x_i)
#
psi = np.zeros(N + 1)

i1, i2 = int(x1 / dx), int(x2 / dx) 
for i in range(i1, i2 + 1):
   psi[i + 1] = 1.         # set psi = 1

normalize(psi, xvals)      # then normalize

dpsi = 1. / sqrt(x2 - x1) * dpsiRel

# set up graphics/plots
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 3

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()

# compute potential for plotting
# x = 0 skipped to avoid 1/0
Nsamples = 1000
xvalsPot = np.zeros(Nsamples + 1)
yvalsPot = np.zeros(Nsamples + 1)
r = 1. / epsilon   # scale epsilon out from potential curve
for i in range(1, Nsamples + 1): 
   x = i * xmax / Nsamples
   xvalsPot[i] = x
   yvalsPot[i] = r * calculateV(x, epsilon, sigma)


# interactive loop
#

print("Type: (c)lear, (q)uit, (h)alve dpsi, (d)ouble dpsi, (t) adjust T for annealing")
print("      anything else generates a new random update")


E = computeE(psi, xvals, epsilon, sigma)  # initial energy

first = True
Ntotal, NtotalAcc, Naccepted = 0, 0, 0    # update success bookkeeping

while True:

   # update parameters
   params = (epsilon, sigma, dpsi, T)

   # on first iteration, set range and fontsize
   if first:
      ax0.tick_params(labelsize = fontsize)
      ax0.set_xlim([0., xmax])
      ax0.set_ylim([-ymax, ymax])

   # on subsequent iterations
   else:
      # replot prev curve in green
      ax0.plot(xvals, psi, color = "green")
      # do an update
      E, Naccepted = update(Nattempts, E, psi, xvals, params)
      Ntotal += Nattempts
      NtotalAcc += Naccepted

   # plot curve
   ax0.plot(xvalsPot, np.zeros(Nsamples + 1), color = "black")   # psi = 0 baseline
   ax0.plot(xvalsPot, yvalsPot, color = "blue")      # potential
   ax0.plot(xvals, psi, color = "red")            # psi

   titleStr = ("E=%0.5e, dpsi=%0.5e, T=%0.5e" % (E, dpsi, T))
   titleStr += "\n" + ("accepted=%d, total att=%d, total acc=%d" % (Naccepted, Ntotal, NtotalAcc))
   ax0.set_title(titleStr)

   plt.pause(0.001)
   first = False

   # get user input
   inp = input("(c)lear, (q)uit, (h)alve dpsi, (d)ouble dpsi, (t) rescale T: ")

   if inp == "q": break
   elif inp == "c":
      ax0.clear()
      first = True
   elif inp == "h":
      dpsi *= 0.5
   elif inp == "d": 
      dpsi *= 2.
   elif inp == "t":
      if T > 0.: 
         alpha = float( input("  lower T by fraction: ")  )
         T *= (1. - alpha)


# write output file
if fname != "":
   with open(fname, "w+") as f:
      for i in range(N + 1):
         f.write( str(xvals[i]) + " " + str(psi[i]) + "\n" ) 


#EOF
