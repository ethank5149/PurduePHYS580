#
# Cannon ball trajectory computed via Euler approximation
#
# assumes: i) drag force F = - B_2 * v^2
#          ii) isothermal air density model: rho(y) = rho(0) * exp(-y / y0)
#
# by Denes Molnar for PHYS 580, Fall 2020

import math
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# computes arrays of x,y for parameters v0, theta, B2/m, y0scale, dt
#
def computeTrajectory(v0 = 700., theta = 45., B2_m = 4e-5, y0scale = 1e4, dt = 0.01):
   # convert to Cartesian initconds
   x0 = 0.
   y0 = 0.
   vx = v0 * math.cos(theta / 180. * math.pi);
   vy = v0 * math.sin(theta / 180. * math.pi);

   maxr = v0 * v0 / g   # max range of shell, in vaccuum (for automatic horizontal plot range)
   maxt = maxr / vx     # flight time at maxr, in vacuum (for atomatic calculation end time)
   nsteps = math.ceil(maxt / dt)   # time steps

   # allocate storage
   # - only position is stored (needed for plotting)
   x = np.zeros(nsteps + 1)
   y = np.zeros(nsteps + 1)

   # solve using Euler method
   for i in range(nsteps):
      x[i + 1] = x[i] + vx * dt
      y[i + 1] = y[i] + vy * dt
      v = math.sqrt(vx**2 + vy**2)
      fD_over_v = B2_m * v * math.exp(-y[i] / y0scale)    # isothermal model, hardcoded
      vx = vx -      fD_over_v * vx  * dt
      vy = vy - (g + fD_over_v * vy) * dt
      if y[i + 1] <= 0.:   # stop calculation at step (i+1) if y_i is below ground
         break

   # adjust last point to match interpolated range
   xmax = (y[i + 1] * x[i] - y[i] * x[i + 1] ) / (y[i + 1] - y[i])
   x[i + 1] = xmax  # adjust  last point to keep y >= 0 in plots
   y[i + 1] = 0.

   # return appropriately truncated arrays
   return x[:(i+2)], y[:(i+2)]



# physical parameters and initial conditions (units are in [] brackets)
#
#   v0 [m/s], theta [degrees], B2/m  [1/m],  y0 [m]
#
# some reasonable choices are B2/m = 4x10^(-5), y0 = 10^4
#
v0      = float( input('initial speed [m/s]: ') )
theta   = float( input('shooting angle [degrees]: ') )
B2_m    = float( input('B2/m [1/m]: ') )
y0scale = float( input('characteristic height in atmospheric pressure [m]: ') ) 
g       = 9.8   # gravitational acceleration [m/s^2]

# computational parameters: dt[s], nsteps (nsteps is autodetermined)
dt = float( input('dt [s]: ') )

# compute trajectory with Euler approximation
x,y = computeTrajectory(v0, theta, B2_m, y0scale, dt)

# y vs x plot
plt.xlabel('x [m]')    # horizontal position
plt.ylabel('y [m]')    # vertical position
titleString  = 'dt=' + str(dt) + ", v0=" + str(v0) + ", B2/m=" + str(B2_m)
titleString += ", h=" + str(y0scale) + ", theta=" + str(theta) 
plt.title(titleString) # plot title
plt.plot(x, y, label = "Euler")
plt.legend()             # create legends
plt.show()               # show plot


#EOF
