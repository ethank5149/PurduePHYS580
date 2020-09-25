# computes trajectories under central gravitational force
# 
# solves 
#
#    d^2r/dt^2 = - G*M r / |r|^(beta + 1) 
#
# in the x-y plane from given initial conditions using the Euler-Cromer method
#
# AU-year units throughout (for Solar System)
#
#
# by Denes Molnar for PHYS 580, Fall 2020


from math import sqrt
from math import pi as PI
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# compute evolution from given initconds
def computeTrajectory(initconds, dt, nsteps):
   (x0, y0, vx0, vy0) = initconds

   # allocate storage - only store position
   x = np.zeros(nsteps + 1)
   y = np.zeros(nsteps + 1)
   x[0], y[0] = x0, y0

   vx, vy = vx0, vy0  # only store instantaneous velocity

   # Euler-Comer method
   GM = 4. * PI**2
   for i in range(nsteps):
      # update velocity first
      r = sqrt(x[i]**2 + y[i]**2)
      vx = vx - GM * x[i] * dt / r**3
      vy = vy - GM * y[i] * dt / r**3
      # then position
      x[i + 1] = x[i] + dt * vx
      y[i + 1] = y[i] + dt * vy

   # return results
   return x, y


# get parameters
#

# initial conditions
x0  = float( input('x_0 [AU]: ') )
y0  = float( input('y_0 [AU]: ') )
vx0 = float( input('vx_0 [AU/yr]: ') )
vy0 = float( input('vy_0 [AU/yr]: ') )

# computational parameters
t_end  = float( input('t_end [yr]: ') )
dt     = float( input('dt [yr]: ') )

nsteps = int(t_end / dt + 0.5)


# first compute entire evolution
#

initconds = (x0, y0, vx0, vy0)

xvals, yvals = computeTrajectory(initconds, dt, nsteps)


# plot trajectory as an "animation"
#

#fontsize = 30   # fontsize, need to be tuned to screen resolution
fontsize = 15    # fontsize, for 1920x1080
linewidth = 1.5
pointarea = 2.

# make 1 plot
fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

ax0.tick_params(labelsize = fontsize)
#ax0.legend(fontsize = fontsize)
ax0.set_xlabel("x [AU]", fontsize = fontsize)
ax0.set_ylabel("y [AU]", fontsize = fontsize)
ax0.set_xlim((-1.5, 1.5))
ax0.set_ylim((-1.5, 1.5))

# depict Sun at origin
ax0.add_artist( plt.Circle((0, 0), 0.2, color = "orange") ) 

# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()

# start by plotting an empty curve, then replot with more and more data
curves = ax0.plot([], [],  linewidth = linewidth, color = "blue")

batchsize = 10
for i in range(0, nsteps, batchsize):
   iend = i + batchsize

   # OPTION 1 (rudimentary): add new points via scatter
   #   - points are unconnected
   #   - each batch is a new dataset, technically, so must control color
   #ax0.scatter(xvals[i:iend], yvals[i:iend],  s = pointarea, color = "blue")

   # OPTION 2 (simplest): plot a new segment with each update
   #   - small amount of data duplication at segment boundaries
   #   - each batch is a new curve, technically, so must control color
   ax0.plot(xvals[i:(iend + 1)], yvals[i:(iend + 1)],  linewidth = linewidth, color = "blue")

   # OPTION 3: update underlying curve data
   #   - keeps data in contiguous chunk, one curve
   #   - longer code
   #xdata = curves[0].get_xdata()
   #ydata = curves[0].get_ydata()
   #xdata = np.append(xdata, xvals[i:(i + batchsize)])
   #ydata = np.append(ydata, yvals[i:(i + batchsize)])
   #curves[0].set_xdata(xdata)
   #curves[0].set_ydata(ydata)

   # OPTION 4 (professional): use pyplot animation
   #   - ultimate flexibility
   #   - additional abstraction, yet longer code
   #
   # see https://matplotlib.org/api/animation_api.html#module-matplotlib.animation
   # [the example code for FuncAnimation is especially useful there]

   year = round(dt * iend, 3)  # avoid long numbers due to round-off errors
   titleString  = "dt=" + str(dt) + ", t=" + str(year) + " [yr]"
   ax0.set_title(titleString, fontsize = fontsize)

   plt.pause(0.2)  # update plot window and sleep for some time


# DONE, wait for user input before closing the plot window
input("End [ENTER]")


#EOF
