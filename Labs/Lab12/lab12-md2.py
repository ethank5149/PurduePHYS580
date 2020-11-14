# molecular dynamics in 2D
#
# - Lennard-Jones potential
# - reduced (dimensionless) quantities
# - LxL box with periodic boundary conditions
# - initconds: random velocity
#              particles on a grid + a random displacement
#
#
# by Denes Molnar for PHYS 580, Fall 2020



import numpy as np
from random import random
from math import sqrt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# sign function
def sgn(x):
   if x < 0.:    return -1.
   elif x > 0.:  return 1.
   else:         return 0.


# compute one timestep
def update(r, rprev, params):
   # unpack params
   (N, L, dt, rcut) = params

   # storage for updated position and current velocity
   rnew = np.zeros( (N, 2) )
   v = np.zeros( (N, 2) )

   # Verlet update
   for i in range(N):
      # compute total force
      fxTot, fyTot = 0., 0.
      for j in range(N):
         if j == i: continue   # no self-interaction
         # relative position - take nearest separation (periodic BC)
         xij = r[i,0] - r[j,0]
         yij = r[i,1] - r[j,1]
         if abs(xij) > L * 0.5:
           xij -= sgn(xij) * L
         if abs(yij) > L * 0.5:
           yij -= sgn(yij) * L
         rij = sqrt(xij**2 + yij**2)
         # skip r > rcut
         if rij > rcut: continue
         # force j->i
         fjiMagnitude = (48. / rij**13 - 24. / rij**7)
         fxTot += fjiMagnitude * xij / rij
         fyTot += fjiMagnitude * yij / rij
      # new positions and velocities from Verlet
      xnew = 2. * r[i,0] - rprev[i,0] + dt**2 * fxTot
      ynew = 2. * r[i,1] - rprev[i,1] + dt**2 * fyTot
      v[i,0] = (xnew - rprev[i,0]) / (2. * dt)
      v[i,1] = (ynew - rprev[i,1]) / (2. * dt)
      # employ periodic BC, then store position
      if   xnew < 0.:
         xnew += L
         r[i,0] += L
      elif xnew > L:
         xnew -= L
         r[i,0] -= L
      if   ynew < 0.:
         ynew += L
         r[i,1] += L
      elif ynew > L:
         ynew -= L
         r[i,1] -= L
      rnew[i,0] = xnew
      rnew[i,1] = ynew
   # return new position, current position, and current velocity arrays
   return rnew, r, v


# calculate energy, and estimate temperature
def calculateE(r, v, params):

   (N, L, _, rcut) = params

   Ekin, Epot = 0., 0.
   for i in range(N):
      # add kinetic energy
      Ekin += 0.5 * (v[i,0]**2 + v[i,1]**2)
      # for potential energy, take each pair once (e.g., j < i)
      for j in range(i):
         xij = r[i,0] - r[j,0]
         yij = r[i,1] - r[j,1]
         if abs(xij) > L * 0.5:
           xij -= sgn(xij) * L
         if abs(yij) > L * 0.5:
           yij -= sgn(yij) * L
         rij = sqrt(xij**2 + yij**2)
         # ignore interactions for r > rcut
         if rij > rcut: continue
         # add contribution
         invr6 = 1. / rij**6
         invr12 = invr6 * invr6
         Epot += 4. * (invr12 - invr6)
   # return 1) total energy, and 2) temperature estimate T = Ekin / N
   return Ekin + Epot, Ekin / N


# plot particles, highlight 1st and 2nd particle
def plotParticles(ax0, r, L):
   ax0.set_xlim( [0, L] )
   ax0.set_ylim( [0, L] )
   #
   ax0.scatter(r[:,0], r[:,1], s = pointarea, color = "red")     # used red, except
   ax0.scatter(r[0,0], r[0,1], s = pointarea, color = "blue")    # particle 1 is in blue 
   ax0.scatter(r[1,0], r[1,1], s = pointarea, color = "green")   # particle 2 is in green






# read parameters
#
N    = int(   input('Number of particles: ') )
L    = float( input('Side length L of box [L/sigma units]: ') )
dt   = float( input('Time step [reduced units, see p.274]: ') )
plotFreq = int(   input('Number of timesteps between plots/output: ') )
vmax = float( input('Max initial speed [reduced units]: ') )
dmax = float( input('Max initial displacement rel. to grid sites: ') )
fname  = input('output file name: ').strip()

rcut = 3.   # cutoff for interaction: r > 3

params = (N, L, dt, rcut)


# open output file (truncate)
# 
if fname != "":   f = open(fname, "w+")
else:             f = None


# initial conditions
#

r     = np.zeros( (N, 2) )   # current position
rprev = np.zeros( (N, 2) )   # prev position
v     = np.zeros( (N, 2) )   # current velocity

# arrange particles on a rectangular grid, to keep them apart initially
# - use smallest grid that can house them
# - grid only partially filled if N is not square
Ngrid = int(sqrt(N))
if Ngrid != sqrt(N):
   Ngrid += 1
Lgrid = L / Ngrid

for k in range(N):
   # row, column and corresponding position (sqrt(2) factors maintain parity with old Matlab version)
   i, j = k // Ngrid,  k % Ngrid
   r[k,0] = (i + 0.5) * Lgrid + (random() - 0.5) * dmax * Lgrid * sqrt(2.)
   r[k,1] = (j + 0.5) * Lgrid + (random() - 0.5) * dmax * Lgrid * sqrt(2.)
   # velocity
   v[k,0] = vmax * (random() - 0.5) * sqrt(2.)
   v[k,1] = vmax * (random() - 0.5) * sqrt(2.)
   # previous position (assumes no interaction before t = 0)
   rprev[k,0] = r[k,0] - v[k,0] * dt
   rprev[k,1] = r[k,1] - v[k,1] * dt



# display initial configuration
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 3
pointarea = 80. * (20. / N)**2   # scale to N = 20 

fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

ax0.tick_params(labelsize = fontsize)
plotParticles(ax0, r, L)


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


plt.pause(0.001)
input('[ENTER]')


# interactive loop
# - also records physics output

print("Type: 'c' to clear tracks")
print("      'd' to clear and disable green tracks")
print("      'u' to unable green tracks")
print("      'q' to move on to analysis plots")
print("      '+/-' to decreas velocities by +50%/-50%")
print("      '1/2/3/4' to increase velocities by 10, 20, 30, 40%")
print("      anything else for another batch of position updates")

output = []

t = 0.
doPlot = True

while True:

   # compute next plotFreq timesteps
   for _ in range(plotFreq):
      r, rprev, v = update(r, rprev, params)
   t += plotFreq * dt
   # physics output
   # - energy, temperature at now previous timestep
   E, T = calculateE(rprev, v, params)
   # - |r|^2 for particle 1, and |r1-r2|^2 for particles 1 & 2
   r1sq  = rprev[0,0]**2 + rprev[0,1]**2
   r12sq = (rprev[0,0] - rprev[1,0])**2 + (rprev[0,1] - rprev[1,1])**2
   output.append( [t, E, T, r1sq, r12sq] )
   if f != None:
      f.write(str(t) + " " + str(E) + " " + str(T) + " " + str(r1sq) + " " + str(r12sq) + "\n")
      f.flush()
   # plot positions
   if doPlot:
      ax0.scatter(r[:,0], r[:,1], s = pointarea, color = "green", facecolors = "none", linewidth = 1)
      titleStr = "t=" + str(round(t, 4)) + ", E=" + str(round(E,4)) + ", T=" + str(round(T,4))
      ax0.set_title(titleStr, fontsize = fontsize)
      plt.pause(0.001)

   # user input on what to do next
   inp = input("input [q,c,d,u,+,-,1,2,3,4]: ").strip()
   scale = None

   if inp == "q":
      break
   elif inp == "c" or inp == "d":
      ax0.clear()
      plotParticles(ax0, r, L)
      plt.pause(0.001)
      if inp == "d":   doPlot = False
   elif inp == "u":    doPlot = True
   elif inp == "+":    scale = 1.5
   elif inp == "-":    scale = 0.5
   elif inp == "1":    scale = 1.1
   elif inp == "2":    scale = 1.2
   elif inp == "3":    scale = 1.3
   elif inp == "4":    scale = 1.4

   # scale velocities upon request
   if scale != None:
      for k in range(N):
         rprev[k,0] = r[k,0] - scale * (r[k,0] - rprev[k,0])
         rprev[k,1] = r[k,1] - scale * (r[k,1] - rprev[k,1])



# close file
if f != None:  f.close()


# time series plots
#

output = np.array(output)

# E vs t
ax0.clear()
ax0.set_xlabel('t [red. units]', fontsize = fontsize)
ax0.set_ylabel('E [red. units]', fontsize = fontsize)
ax0.set_title('Energy vs time', fontsize = fontsize)
ax0.plot(output[:,0], output[:,1], linewidth = linewidth, color = "blue")

plt.pause(0.01)
input("[ENTER]")

# T vs t
ax0.clear()
ax0.set_xlabel('t [red. units]', fontsize = fontsize)
ax0.set_ylabel('T [red. units]', fontsize = fontsize)
ax0.set_title('Temperature (estimated) vs time', fontsize = fontsize)
ax0.plot(output[:,0], output[:,2], linewidth = linewidth, color = "blue")

plt.pause(0.01)
input("[ENTER]")

# |r1|^2 vs t
ax0.clear()
ax0.set_xlabel('t [red. units]', fontsize = fontsize)
ax0.set_ylabel('|r1|^2 [red. units]', fontsize = fontsize)
ax0.set_title('Position-squared vs time', fontsize = fontsize)
ax0.plot(output[:,0], output[:,3], linewidth = linewidth, color = "blue")

plt.pause(0.01)
input("[ENTER]")

# |r1-r2|^2 vs t
ax0.clear()
ax0.set_xlabel('t [red. units]', fontsize = fontsize)
ax0.set_ylabel('|r1-r2|^2 [red. units]', fontsize = fontsize)
ax0.set_title('Relative distance squared vs time', fontsize = fontsize)
ax0.plot(output[:,0], output[:,4], linewidth = linewidth, color = "blue")

plt.pause(0.01)
input("[ENTER]")

#EOF
