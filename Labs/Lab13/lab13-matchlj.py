# time-independent Schroedinger solutions in 1D via the 'shooting' method
#
# - Lennard-Jones potential
# - m=1, hbar = 1, L = 1 units (same as in the Giordano-Nakanishi textbook)
# - psi is taken to be real (without loss of generality)
# - shoots to right from xleft = 0.5*sigma to right, and to left from xright = 5*sigma,
#   and tracks matching at xmatch = 1.4 * sigma
# - agreement in psi(x) at xmatch enforced via scaling, so only remaining question 
#   is whether derivatives also match there
#
#	
# NOTE: does not unit normalize psi, but that is unnecessary (could be done later)
#
#
# by Denes Molnar for PHYS 580, Fall 2020



import numpy as np
from math import sqrt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Lennard-Jones potential
def calculateV(x, eps, sigma):
   invr6 = (sigma / x)**6
   return 4. * eps * (invr6 * invr6 - invr6)


# compute solution and derivative at endpoint for given grid (equidistant grid ASSUMED)
# 
# solution is scaled to psi = 1 at end point
#
# initial values are in psi[0] and psi[1]
# 
def computePsi(E, xvals, psi, params):

   # unpack params
   (eps, sigma, Nextra) = params

   dx = xvals[1] - xvals[0]
   N = xvals.shape[0] - 1 - Nextra


   # compute psi(x) - basically just like Verlet update
   for i in range(1, N + Nextra):
      x = xvals[i]
      V = calculateV(x, eps, sigma)
      psi[i + 1] = 2. * psi[i] - psi[i - 1] - 2. * (E - V) * dx**2 * psi[i]

   # normalize solution to psi = 1 at endpoint
   fact = 1. / psi[N]
   psi *= fact

   # estimate derivative at end, and return it
   psiPrime = 0.5 * (psi[N + 1] - psi[N - 1]) / dx
   return psiPrime



# read parameters
#

print( "Single particle in 1D Lennard-Jones potential:" )
epsilon  = float( input('  L-J energy scale [hbar^2/(m L^2)], epsilon= ') )
sigma    = float( input('  L-J length scale [units of L], sigma= ') )

# computational params
print( "Calculation in box 0 <= x <= 5*sigma [in units of L]:" )
dx       = float( input('  grid cell size [units of L], dx= ') )
E        = float( input('  initial guess for energy E= ') )
dE       = float( input('  initial energy increment dE= ') )

# hardcoded values
#
# matching location for left & right solutions
xmatch = 1.4 * sigma
# starting locations for left & right solutions
# - slightly readjusted later to fall exactly on grid sites
xleft  = 0.5 * sigma
xright = 5. * sigma
# extra sites to plot beyond matching point (MUST be at least 1)
Nextra = 16
# yrange for plots [-ymax, ymax]
ymax = 2.

# shorthands for psi(x) calculator
params = (epsilon, sigma, Nextra)


# set up grids
# 
Nleft = int((xmatch - xleft) / dx) + 1
xleft = xmatch - dx * Nleft
Nright = int((xright - xmatch) / dx) + 1
xright = xmatch + dx * Nright

xvalsL = np.zeros(Nleft + 1 + Nextra)
xvalsR = np.zeros(Nright + 1 + Nextra)
for i in range(Nleft + 1 + Nextra):
   xvalsL[i] = xmatch - (Nleft - i) * dx
for i in range(Nright + 1 + Nextra):
   xvalsR[i] = xmatch + (Nright - i) * dx      # decreasing order in x(!)

psiL = np.zeros(Nleft + 1 + Nextra)
psiR = np.zeros(Nright + 1 + Nextra)

# initconds" for shooting: zero values, nearly flat slopes
psiL[0] = 0.
psiL[1] = 1e-2
psiR[0] = 0.
psiR[1] = 1e-2


# set up graphics/plots
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 3

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()

# compute potential for plotting, rescaled vertically by 1/epsilon
Nsamples = 1000
xvalsPot = np.zeros(Nsamples + 1)
yvalsPot = np.zeros(Nsamples + 1)
r = 1. / epsilon
for i in range(Nsamples + 1):
   x = xleft + i * (xright - xleft) / Nsamples
   xvalsPot[i] = x
   yvalsPot[i] = r * calculateV(x, epsilon, sigma)


# interactive loop
#

print("Type: (c)lear, (q)uit, (h)alve dE, (d)ouble dE, (r)everse & halve dE")
print("      anything else just increments/decrements E" )


first = True
prevMatch = 0
while True:

   # replot previous curve in green
   if not first:
      ax0.plot(xvalsL[1:], psiL[1:], color = "green")  # psiL
      ax0.plot(xvalsR[1:], psiR[1:], color = "green")  # psiR


   # compute shooting solutions and derivatives
   psiPrimeL = computePsi(E, xvalsL, psiL, params)
   psiPrimeR = computePsi(E, xvalsR, psiR, params)

   # track sign of difference in derivatives, and change in sign
   match = 1   if  psiPrimeL >= psiPrimeR   else   -1
   mm = match * prevMatch

   # plot it
   if first:
      ax0.tick_params(labelsize = fontsize)
      ax0.set_xlim([0., xright])
      ax0.set_ylim([-ymax, ymax])

   ax0.plot(xvalsPot, np.zeros(Nsamples + 1), color = "black")   # psi = 0 baseline
   ax0.plot(xvalsPot, yvalsPot, color = "blue")      # potential
   ax0.plot(xvalsL[1:], psiL[1:], color = "red")  # psiL
   ax0.plot(xvalsR[1:], psiR[1:], color = "red")  # psiR

   titleStr = "E=" + str(round(E, 5)) + (", dE=%0.5e" % dE) + ", dpsiL/dx=" + str(round(psiPrimeL, 5)) + ", dpsiR/dx=" + str(round(psiPrimeR, 5)) + ", mm=" + str(mm)
   ax0.set_title(titleStr)

   plt.pause(0.001)
   first = False

   # get user input
   inp = input("(c)lear, (q)quit, (h)alve dE, (d)ouble dE, (r)everse & halve dE: ")

   if inp == "q": break
   elif inp == "a":  # automatic choice
      # fill this in for testing (mm can be handy)
      # then make it automatic without keyboard input
      pass
   elif inp == "c":
      ax0.clear()
      first = True
   elif inp == "h":
      dE *= 0.5
   elif inp == "d": 
      dE *= 2.
   elif inp == "r":
      dE *= -0.5

   # update energy, update derivative difference tracking
   E += dE
   prevMatch = match

#EOF
