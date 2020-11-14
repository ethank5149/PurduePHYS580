# computes trajectories of two planets "Earth" and "Jupiter" orbitting the same "Sun",
# in the same orbital plane
# 
# solves coupled set of equations
#
#    d^2r/dt^2 = - G*M r / |r|^(beta + 1) 
#
# in the x-y plane from given initial conditions using the Euler-Cromer method
#
# AU-year units throughout (Solar System)
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
def computeTrajectory(initconds, params, dt, nsteps):
   (x0_E, y0E, vx0_E, vy0_E) = initconds[0]
   (x0_J, y0J, vx0_J, vy0_J) = initconds[1]

   (mE, mJ) = params    # m_i / M_Sun ratios

   # allocate storage - only store positions
   xE, yE = np.zeros(nsteps + 1), np.zeros(nsteps + 1)
   xJ, yJ = np.zeros(nsteps + 1), np.zeros(nsteps + 1)
   xE[0], yE[0] = x0_E, y0_E
   xJ[0], yJ[0] = x0_J, y0_J

   vx_E, vy_E = vx0_E, vy0_E  # only track instantaneous velocity
   vx_J, vy_J = vx0_J, vy0_J

   # Euler-Comer method
   GM = 4. * PI**2
   for i in range(nsteps):
      # distances
      rES = sqrt(xE[i]**2 + yE[i]**2)
      rJS = sqrt(xJ[i]**2 + yJ[i]**2)
      rEJ = sqrt( (xE[i] - xJ[i])**2 + (yE[i] - yJ[i])**2 )
      # update velocities first
      vx_E = vx_E + dt * (- GM * xE[i] / rES**3 - GM*mJ * (xE[i] - xJ[i]) / rEJ**3)
      vy_E = vy_E + dt * (- GM * yE[i] / rES**3 - GM*mJ * (yE[i] - yJ[i]) / rEJ**3)
      vx_J = vx_J + dt * (- GM * xJ[i] / rJS**3 - GM*mE * (xJ[i] - xE[i]) / rEJ**3)
      vy_J = vy_J + dt * (- GM * yJ[i] / rJS**3 - GM*mE * (yJ[i] - yE[i]) / rEJ**3)
      # then positions
      xE[i + 1] = xE[i] + dt * vx_E
      yE[i + 1] = yE[i] + dt * vy_E
      xJ[i + 1] = xJ[i] + dt * vx_J
      yJ[i + 1] = yJ[i] + dt * vy_J

   # return results
   return xE, yE, xJ, yJ


# get parameters
#

# initial conditions
print("Planet 1:")
m_E   = float( input('  m_1 / M_star: ') )
x0_E  = float( input('  x_0 [AU]: ') )
y0_E  = float( input('  y_0 [AU]: ') )
vx0_E = float( input('  vx_0 [AU/yr]: ') )
vy0_E = float( input('  vy_0 [AU/yr]: ') )

print("Planet 2:")
m_J   = float( input('  m_2 / M_star: ') )
x0_J  = float( input('  x_0 [AU]: ') )
y0_J  = float( input('  y_0 [AU]: ') )
vx0_J = float( input('  vx_0 [AU/yr]: ') )
vy0_J = float( input('  vy_0 [AU/yr]: ') )


# computational parameters
t_end  = float( input('t_end [yr]: ') )
dt     = float( input('dt [yr]: ') )

nsteps = int(t_end / dt + 0.5)


# first compute entire evolution
#

initconds = ((x0_E, y0_E, vx0_E, vy0_E), (x0_J, y0_J, vx0_J, vy0_J))
params = (m_E, m_J)

xEvals, yEvals, xJvals, yJvals = computeTrajectory(initconds, params, dt, nsteps)


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
ax0.set_xlim( (-7, 7) )
ax0.set_ylim( (-7, 7) )

# depict Sun at origin
ax0.add_artist( plt.Circle((0, 0), 0.2, color = "orange") ) 

# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()


# plot by segments of a given size
batchsize = 10
for i in range(0, nsteps, batchsize):
   iend = i + batchsize

   # plot next segment for each planet
   ax0.plot(xEvals[i:(iend + 1)], yEvals[i:(iend + 1)],  linewidth = linewidth, color = "blue")
   ax0.plot(xJvals[i:(iend + 1)], yJvals[i:(iend + 1)],  linewidth = linewidth, color = "red")

   # update plot title 
   year = round(dt * iend, 3)  #avoid long numbers due to round-off errors
   titleString  = "star fixed, 2 planets: dt=" + str(dt) + ", t=" + str(year) + " [yr]"
   ax0.set_title(titleString, fontsize = fontsize)

   plt.pause(0.01)  # update plot window and sleep for some time


# DONE, wait for user input before closing the plot window
input("End [ENTER]")


#EOF
