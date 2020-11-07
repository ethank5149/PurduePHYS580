# Jacobi method to solve Laplace equation for a parallel-plate capacitor
# (Sec 5.1 of the Giordano-Nakanishi 'Computational Physics' book)
#
# capacitor geometry: 
#   - infinite extension in z direction
#   - 2D cut at z = const perpendicular to plates:  -L/2 < x,y < L/2
#   - plates parallel to y axis, plate width (y direction) is a*L
#   - separation distance b*L in x direction
#
# boundary conditions:
#   - V = 0 at the outside boundaries of the box
#   - V = -V0 at x = -L*b/2,  +V0 at x = +L*b/2 
#


import numpy as np
from math import floor


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Jacobi update for V
# - returns new V and the average of per-site changes in V (convergence measure)
def updateJacobi(V):
  # infer grid dimensions
  (nx, ny) = V.shape
  # allocate new array
  Vnew = np.zeros( (nx, ny) )
  # move along the grid, element by element
  # - skip box boundaries -> preserves zeroes there
  # - on plate, copy old values -> plates stay same
  deltaV = 0.
  for i in range(1, nx - 1):
    for j in range(1, ny - 1):
      # on the plates, copy
      if V[i,j] == 1. or V[i,j] == -1.:   # "cheap" way to detect plates
         Vnew[i,j] = V[i,j]
      # update otherwise
      else:
         Vnew[i,j] = 0.25 * (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1])
         deltaV = deltaV + abs(Vnew[i,j] - V[i,j])
  # convert deltaV to per-lattice-site error
  deltaV /= nx * ny
  #
  return Vnew, deltaV


# read parameters
#
# geometry
a = float( input('width of plates (a) in units of L: ') )
b = float( input('plate separation (b) in units of L: ') )

# computational params
dL  = float( input('grid cell size (dx = dy = dL) in units of L: ') )
acc = float( input('acceptable local error as fraction of V0: ') )


# real (x,y) geometry <=> grid [i,j] 
#
# lowest corner: [0,0] = V(-L/2, -L/2) 
# center:        [M,M] = V(0,0) 
#
M = floor(0.5 / dL + 0.5)  # number of sites along half length of box (rounded)
dL = 1. / (2 * M)        # readjust dL to match grid perfectly
N = 2 * M  + 1           # number of grid points along one direction (incl. edges)
aM = floor(a*M + 0.5)    # half widths of plates, rounded to nearest gridpoint
bM = floor(b*M + 0.5)    # half separation of plates, rounded to nearest gridpoint

# initialization (zeros everywhere + BC)
V = np.zeros( (N, N) )
for i in range(-aM, aM + 1):
    V[M - bM, M + i] = -1.
    V[M + bM, M + i] = +1.


# prepare 1 figure (3D)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create an x-y mesh (X & Y are 2D arrays of x and y values)
#
xvals = [ -0.5 + i * dL  for i in range(N) ]
yvals = [ -0.5 + i * dL  for i in range(N) ]
(Y, X) = np.meshgrid(xvals, yvals)   #Y,X order!

# interactive mode, create plot window
matplotlib.pyplot.ion() 
plt.show()

# iteration loop
iter = 0
batchsize = 1  # number of iterations between replots
#batchsize = 20
deltaV = 1e20   # force at least 1 iteration
while deltaV > acc:
    # draw surface
    if (iter % batchsize) == 0:
       ax.clear()  # clear figure 
       surf = ax.plot_surface(X, Y, V, alpha = 0.8)       # surface plot
       titleStr = "Jacobi, iteration " + str(iter)
       if iter > 0: titleStr += ", deltaV=" + str(deltaV);
       ax.set_title(titleStr)
       ax.set_xlabel("x / L")
       ax.set_ylabel("y / L")
       ax.set_zlabel("V / V0")
       plt.draw()
       input("next batch [ENTER]")
    V, deltaV = updateJacobi(V)
    iter = iter + 1

# wait for 'q' to terminate
while input("Finished [q] ") != "q":
   continue


#EOF
