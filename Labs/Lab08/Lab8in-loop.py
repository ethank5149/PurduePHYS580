# field of a loop in the x-y plane, evaluated over a square in the x-z plane
#
# geometry:
# - loop and square are both centered at the origin
# - in units of L, loop radius = R/L, square size = 2 (-L to L in x and z)
# - current in counter-clockwise direction when viewed from a point on the +z axis
#
# units: I * mu0 / 4pi = 1
#
# by Denes Molnar, for PHYS 580 (Fall 2020)
#

import numpy as np
from math import floor, sin, cos, sqrt
from math import pi as PI

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# calculate the B field at position (x,y,z)
def calculateB(x, y, z, R, Nphi):
   # initialize sums for integral
   bx, by, bz = 0., 0., 0.
   # step size for angular integral
   dphi = 2. * PI / Nphi
   # loop through phi elements
   for i in range(Nphi):
      phi = i * dphi
      # components of line element dl (note, dlz = 0)
      dlx = - R * dphi * sin(phi)
      dly =   R * dphi * cos(phi)
      # components of vector from line element to point of observation
      rx = x - R * cos(phi)
      ry = y - R * sin(phi)
      rz = z
      r = sqrt(rx**2 + ry**2 + rz**2)
      # sum contributions dl x r / r^3, but avoid getting close to the wire
      if r > R * 1e-4:
         bx = bx +  dly * rz / r**3
         by = by -  dlx * rz / r**3
         bz = bz + (dlx * ry - dly * rx) / r**3
   return (bx, by, bz)



# read computational parameters
#
# - M: controls the number of x-z plane grid points to store B for
# - Nphi: number of angular points
#   (minimal setting might be M=10, Nphi=25)
#
print('Field of loop in x-y plane, evaluated over an x-z-plane square of size 2L')
R = float( input('loop radius (R) in units of L: ') )
M = int( input('grid cells from ORIGIN to edge of rectangle (M): ') )
Nphi = int( input('phi steps along loop (Nphi): ') )


# calculate field by sweeping through the xz plane here
#
# - real position (x,z)       <-> grid site [i,k]
# - origin        (0,0)       <->           [M,M]
# - lowest corner (-L,-L)     <->           [0,0]

N = 2 * M + 1
bxVals = np.zeros( (N, N) )
byVals = np.zeros( (N, N) )
bzVals = np.zeros( (N, N) )

dx = 2. / (2 * M)
for i  in range(N):       # x direction
   for k in range(N):     # z direction
      x = dx * (i - M)
      z = dx * (k - M)
      # calculate B at each observation point in x-z plane
      (bx, by, bz) = calculateB(x, 0., z, R, Nphi)
      # store result - NOTE: (i,k) order, i.e., (x,z)
      bxVals[i,k] = bx
      byVals[i,k] = by
      bzVals[i,k] = bz


# plot the B vector field in xz-plane
#

#fontsize = 30   # fontsize, need to be tuned to screen resolution
fontsize = 15    # fontsize, for 1920x1080
linewidth = 1.5
#linewidth = 3
pointarea = 50.

# prepare 1 figure 2D
fig, axArray = plt.subplots(1, 1)
(ax0) = axArray

xvals = [ i * dx  for i in range(-M, M + 1) ]
zvals = [ i * dx  for i in range(-M, M + 1) ]
[Z, X] = np.meshgrid(xvals, zvals)   #Z,X order!


# plot
ax0.quiver(X, Z, bxVals, bzVals, linewidth = linewidth, color = "blue")
ax0.set_xlim((-1, 1))
ax0.set_ylim((-1, 1))
ax0.set_title("Angular intervals = " + str(Nphi));
ax0.set_xlabel("X / L")
ax0.set_ylabel("Z / L")


# turn on interactive mode, so plt.show() does not block, then create plot window
matplotlib.pyplot.ion()
plt.show()

input("[ENTER]")

# draw the loops
ax0.add_artist( plt.Circle((+R, 0), 0.05, color = "orange") )
ax0.add_artist( plt.Circle((-R, 0), 0.05, color = "orange") )

input("[ENTER]")


# plot B_z and B_x vs z
#
ax0.clear()
ax0.plot(zvals, bzVals[M, :], linewidth = linewidth, color = "blue", label = "B_z")
ax0.plot(zvals, bxVals[M, :], linewidth = linewidth, color = "red", label = "B_x")
ax0.set_xlabel('z / L');
ax0.set_ylabel('B_z or B_x');
ax0.set_title('Magnetic field of a current loop');
ax0.legend(fontsize = fontsize)

input("[ENTER]")

# superimpose theoretical result for the loop
# BEWARE: this is _not_ a meaningful test of accuracy...
#
scale = 10
zvalsFine = [ i * dx / scale    for i in range(-M * scale, M * scale + 1) ]

bzExact = np.zeros(len(zvalsFine))
for k,z in enumerate(zvalsFine):
    bzExact[k] = 2. * PI * R**2 / (z**2 + R**2)**1.5

ax0.plot(zvalsFine, bzExact, linewidth = linewidth, color = "orange", label = "exact B_z")
ax0.legend(fontsize = fontsize)

plt.pause(0.01) # redraw


# wait for user input before closing plot window
while input("Finished [q]") != "q":
   pass



#EOF
