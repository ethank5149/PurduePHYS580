#
# undamped pendulum motion, in small-angle approximation
#
# ODE system:   1) dtheta/dt = omega
#               2) domega/dt = -Omega0^2 * theta
#
#     where Omega0^2 = g / L
#
# solved numerically via Euler's method
#
#
# by Denes Molnar for PHYS 580, Fall 2020


import math
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# compute evolution of theta and omega from given initconds
def computeTrajectory(initconds, Omega0, dt, nsteps):
   (th0, om0) = initconds

   # allocate storage - store both theta and omega
   theta = np.zeros(nsteps + 1)
   omega = np.zeros(nsteps + 1)
   theta[0] = th0
   omega[0] = om0

   # solve using Euler method
   for i in range(nsteps):
      theta[i + 1] = theta[i] + omega[i] * dt
      omega[i + 1] = omega[i] - Omega0**2 * theta[i] * dt

   # return results
   return theta, omega


# get parameters
#

# physical parameters and initial conditions (units are in [] brackets)
L    = float( input('length of pendulum [m]: ') )
th0  = float( input('initial angle wrt to vertical [rad]: ') )
om0  = float( input('initial angular velocity [rad/s]: ') )
#
g    = 9.8      # gravitational acceleration [m/s^2] (hardcoded)

Omega0 = math.sqrt(g / L)

# computational parameters: dt[s], nsteps
dt = float( input('dt [s]: ') )
nsteps = int( input('number of time steps to calculate: ') )


# compute evolution
#

theta, omega = computeTrajectory( (th0, om0), Omega0, dt, nsteps)
# time values, for plotting
timevals = np.array( [ dt * i  for i in range(nsteps + 1) ] )
# calculate energy at each timestep
#energy = [ omega[i]**2 + Omega0*Omega0 * theta[i]**2   for i in range(nsteps + 1) ]
energy = omega**2 + Omega0**2 * theta**2  # use numpy by-element array operations

# plots
#

fontsize = 30   # fontsize, need to be tuned to screen resolution
linewidth = 3

# create 3 plots under each other
fig, axArray = plt.subplots(3, sharex = True)

(ax0, ax1, ax2) = axArray

# theta
ax0.plot(timevals, theta, linewidth = linewidth, label = "theta")

titleString  = "dt=" + str(dt) + ", L=" + str(L) 
titleString += ", omega0=" + str(om0) + ", theta0=" + str(th0)

ax0.set_title(titleString, fontsize = fontsize)
ax0.tick_params(labelsize = fontsize)
ax0.legend(fontsize = fontsize)

# omega
ax1.plot(timevals, omega, linewidth = linewidth, label = "omega")

ax1.tick_params(labelsize = fontsize)
ax1.legend(fontsize = fontsize)

# energy
ax2.plot(timevals, energy, linewidth = linewidth, label = "energy")

ax2.tick_params(labelsize = fontsize)
ax2.legend(fontsize = fontsize)
ax2.set_xlabel("t[s]", fontsize = fontsize)

# show plot
plt.show()


#EOF
