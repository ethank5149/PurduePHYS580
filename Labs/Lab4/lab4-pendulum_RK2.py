#
# damped, driven nonlinear pendulum motion
#
# ODE system:   1) dtheta/dt = omega
#               2) domega/dt = -Omega0^2 * sin(theta) - q omega + fD sin(OmegaD t)
#
#     where Omega0^2 = g / L
#
# solved numerically via the 2nd-order Runge-Kutta method
#
#
# by Denes Molnar for PHYS 580, Fall 2020


from math import sin, cos, sqrt, floor
from math import pi as PI
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# alternatively, you could use this function to limit any angle to [-pi,pi)
def chopAngle(th0):
   theta = th0 - floor(th0 / (2. * PI)) * 2. * PI    # chop to [0, 2pi)
   if theta >= PI: theta -= 2. * PI                  # shift to [-pi,pi)
   return theta


# compute evolution of theta and omega from given initconds
def computeTrajectory(initconds, params, dt, nsteps):
   (th0, om0) = initconds
   (Omega0, q, fD, OmegaD) = params

   Omega0sq = Omega0**2

   # allocate storage - store both theta and omega
   theta = np.zeros(nsteps + 1)
   omega = np.zeros(nsteps + 1)
   theta[0], omega[0] = th0, om0

   # solve using 2nd-order Runge-Kutta method
   t = 0
   for i in range(nsteps):
      # halfstep
      om_half = omega[i] + 0.5 * dt * ( -Omega0sq * sin(theta[i]) - q * omega[i] + fD * sin(OmegaD * t) )
      th_half = theta[i] + 0.5 * dt * omega[i]
      t_half = t + 0.5 * dt
      # full step
      omega[i + 1] = omega[i] + dt * ( -Omega0sq * sin(th_half) - q * om_half + fD * sin(OmegaD * t_half) )
      theta[i + 1] = theta[i] + dt * om_half
      t += dt
      # bound theta (or, you could use the chop function -> slower)
      if   theta[i + 1] >= PI:  theta[i + 1] -= 2. * PI
      elif theta[i + 1] < -PI:  theta[i + 1] += 2. * PI 

   # return results
   return theta, omega


# get parameters
#

# physical parameters and initial conditions (units are in [] brackets)
L   = float( input('length of pendulum [m]: ') )
q   = float( input('dissipation coefficient [1/s]: ') )
fD  = float( input('driving coefficient [1/s^2]: ') )
OmegaD = float( input('driving angular frequency [rad/s]: ') )
th0  = float( input('initial angle wrt to vertical [rad]: ') )
om0  = float( input('initial angular velocity [rad/s]: ') )
#
g    = 9.8      # gravitational acceleration [m/s^2] (hardcoded)

Omega0 = sqrt(g / L)

# computational parameters: dt[s], nsteps
dt = float( input('dt [s]: ') )
nsteps = int( input('number of time steps to calculate: ') )


# compute evolution
#
params = (Omega0, q, fD, OmegaD)
initconds = (th0, om0)

theta, omega   = computeTrajectory(initconds, params, dt, nsteps)

# time values, for plotting
timevals = np.array( [ dt * i  for i in range(nsteps + 1) ] )
# calculate 'energy' at each timestep 
energy  = 0.5 * omega**2  + Omega0**2 * (1. - np.cos(theta))   # use numpy vector ops


# plots
#

#fontsize = 30   # fontsize, need to be tuned to screen resolution
fontsize = 15    # fontsize, for 1920x1080
linewidth = 3
pointarea = 2.

# make a 2x2 array of plots 
fig, axArray = plt.subplots(2, 2)

((ax0, ax1), (ax2, ax3)) = axArray

# theta-omega  (TOP LEFT)
#ax0.plot(theta, omega, linewidth = linewidth, label = "phase space")
ax0.scatter(theta, omega, s = pointarea, label = "phase space")

ax0.tick_params(labelsize = fontsize)
ax0.legend(fontsize = fontsize)
ax0.set_xlabel("theta [rad]", fontsize = fontsize)
ax0.set_ylabel("omega [rad/s]", fontsize = fontsize)

titleString  = "dt=" + str(dt) + ", L=" + str(L) 
titleString += ", fD=" + str(fD) + ", OmegaD=" + str(round(OmegaD, 3))
titleString += ", omega0=" + str(om0) + ", theta0=" + str(th0)
ax0.set_title(titleString, fontsize = fontsize)

# omega (TOP RIGHT)
ax1.plot(timevals, omega, linewidth = linewidth, label = "omega")

ax1.tick_params(labelsize = fontsize)
ax1.legend(fontsize = fontsize)
ax1.set_xlabel("t[s]", fontsize = fontsize)
ax1.set_ylabel("omega[rad/s]", fontsize = fontsize)

# theta (BOTTOM LEFT)
ax2.plot(timevals, theta, linewidth = linewidth, label = "theta")

ax2.tick_params(labelsize = fontsize)
ax2.legend(fontsize = fontsize)
ax2.set_xlabel("t[s]", fontsize = fontsize)
ax2.set_ylabel("theta[rad]", fontsize = fontsize)


# energy (BOTTOM RIGHT)
ax3.plot(timevals, energy, linewidth = linewidth, label = "energy")

ax3.tick_params(labelsize = fontsize)
ax3.legend(fontsize = fontsize)
ax3.set_xlabel("t[s]", fontsize = fontsize)
ax3.set_ylabel("energy [a.u.]", fontsize = fontsize)


# show plot
plt.subplots_adjust(wspace = 0.4, hspace = 0.2)
#fig.tight_layout()
plt.show()


#EOF

