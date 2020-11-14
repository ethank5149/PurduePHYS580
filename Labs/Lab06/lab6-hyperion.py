# rotational and orbital  motion of Hyperion, in the dumbbell model introduced
# in the Giordano-Nakanishi textbook
#
# solves
#
#      d^2r/dt^2 = - G*M r / r^3         for the CM motion
#
#      domega/dt = -3 GM / r^5 * [x cos(theta) + y sin(theta)] * [x sin(theta) - y cos(theta)]
#
# the orbital plane and the plane of rotation re both the x-y plane (assumption)
#
# Hyperion units (circular orbit has radius 1 HU, period 1 Hyr)
#
#
# by Denes Molnar for PHYS 580, Fall 2020


from math import sqrt, sin, cos
from math import pi as PI
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# compute evolution from given initconds
def computeMotion(initconds, dt, nsteps):
   (x0, y0, vx0, vy0, th0, om0) = initconds

   # allocate storage - only store position, angle, angular velocity
   x = np.zeros(nsteps + 1)
   y = np.zeros(nsteps + 1)
   theta = np.zeros(nsteps + 1)
   omega = np.zeros(nsteps + 1)
   # load initconds
   x[0], y[0] = x0, y0
   theta[0], omega[0] = th0, om0

   vx, vy = vx0, vy0    # only track instantaneous velocity

   # Euler-Cromer integrator
   GM = 4. * PI**2    # G * M_Saturn in HU-Hyr units
   for i in range(nsteps):
      # update CM velocity
      r = sqrt(x[i]**2 + y[i]**2)
      vx = vx - GM * x[i] * dt / r**3
      vy = vy - GM * y[i] * dt / r**3
      # then CM position
      x[i + 1] = x[i] + dt * vx
      y[i + 1] = y[i] + dt * vy
      # update axis rotational velocity, then angle
      angularTerm  = x[i] * sin(theta[i]) - y[i] * cos(theta[i])
      angularTerm *= x[i] * cos(theta[i]) + y[i] * sin(theta[i])
      omega[i + 1] = omega[i] - dt * 3. * GM / r**5 * angularTerm
      theta[i + 1] = theta[i] + dt * omega[i + 1]
      # map angle to [-pi, pi)
      if theta[i + 1] >= PI:   theta[i + 1] -= 2. * PI
      if theta[i + 1] < -PI:   theta[i + 1] += 2. * PI

   # return results
   return x, y, theta, omega


# get parameters
#

# initial conditions
x0  = float( input('Hyperion CM position x_0 [HU]: ') )
y0  = float( input('                     y_0 [HU]: ') )
vx0 = float( input('    CM velocity vx_0 [HU/Hyr]: ') )
vy0 = float( input('                vy_0 [HU/Hyr]: ') )
th0 = float( input('Axis angle [rad]:' ) )
om0 = float( input('Axis angular velocity [rad/Hyr]:' ) )

# computational parameters
t_end  = float( input('t_end [Hyr]: ') )
dt     = float( input('dt [Hyr]: ') )

nsteps = int(t_end / dt + 0.5)


# first compute entire evolution
#

initconds = (x0, y0, vx0, vy0, th0, om0)

xvals, yvals, thvals, omvals = computeMotion(initconds, dt, nsteps)
# construct time values
timevals = [ dt * i  for i in range(nsteps + 1) ]


# plots
#

#fontsize = 30   # fontsize, need to be tuned to screen resolution
fontsize = 15    # fontsize, for 1920x1080
#linewidth = 1.5
linewidth = 0.7
#pointarea = 2.
pointarea = 0.5

# make 2x2 plot
fig, axArray = plt.subplots(2, 2)
((ax0, ax1), (ax2, ax3)) = axArray


# TOP LEFT: x-y
#
ax0.tick_params(labelsize = fontsize)
#ax0.legend(fontsize = fontsize)
ax0.set_xlabel("x [HU]", fontsize = fontsize)
ax0.set_ylabel("y [HU]", fontsize = fontsize)
ax0.set_xlim((-1.5, 1.5))
ax0.set_ylim((-1.5, 1.5))
ax0.set_title("CM position")

# depict Saturn at origin
ax0.add_artist( plt.Circle((0, 0), 0.2, color = "purple") ) 

ax0.plot(xvals, yvals, linewidth = linewidth, color = "blue")


# TOP RIGHT: theta-omega
#
ax1.tick_params(labelsize = fontsize)
#ax1.legend(fontsize = fontsize)
ax1.set_xlabel("theta [rad]", fontsize = fontsize)
ax1.set_ylabel("omega [rad/Hyr]", fontsize = fontsize)
ax1.set_title("Axis phase portrait")

#ax1.plot(thvals, omvals, linewidth = linewidth, color = "blue")
ax1.scatter(thvals, omvals, s = pointarea, color = "blue", marker = ".")


# BOTTOM LEFT:  theta vs t
#
ax2.tick_params(labelsize = fontsize)
#ax2.legend(fontsize = fontsize)
ax2.set_xlabel("time [Hyr]", fontsize = fontsize)
ax2.set_ylabel("theta [rad]", fontsize = fontsize)
ax2.set_title("Axis orientation")

ax2.plot(timevals, thvals, linewidth = linewidth, color = "blue")


# BOTTOM RIGHT:  omega vs t
#
ax3.tick_params(labelsize = fontsize)
#ax3.legend(fontsize = fontsize)
ax3.set_xlabel("time [Hyr]", fontsize = fontsize)
ax3.set_ylabel("omega [rad/Hyr]", fontsize = fontsize)
ax3.set_title("Angular velocity")

ax3.plot(timevals, omvals, linewidth = linewidth, color = "blue")



# add some spacing between plots
plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
# create plot window and show plot
plt.show()

#EOF
