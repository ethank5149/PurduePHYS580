########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 8                                                                            #
#      \\          |                                                                                                   #
#      //          |  Author: Ethan Knox                                                                               #
#     //           |  Website: https://www.github.com/ethank5149                                                       #
#     ========     |  MIT License                                                                                      #
########################################################################################################################
########################################################################################################################
# License                                                                                                              #
# Copyright 2020 Ethan Knox                                                                                            #
#                                                                                                                      #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated         #
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the  #
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to      #
# permit persons to whom the Software is furnished to do so, subject to the following conditions:                      #
#                                                                                                                      #
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the #
# Software.                                                                                                            #
#                                                                                                                      #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE #
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS  #
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR  #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.     #
########################################################################################################################

# Including
from lib.NDSolveSystem import ODE
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
earth = Constants.Earth()


# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient


def rhs_varying_g(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_ref*v*X[2]/m, -earth.GM/(earth.radius+X[1])**2-B2_ref*v*X[3]/m])


def rhs_constant_g(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_ref*v*X[2]/m, -g-B2_ref*v*X[3]/m])


def terminate(X):
    return X[1] < 0


def deg_to_rad(theta):
    return np.pi*theta/180


x0 = y0 = 0
v0 = 10000
angles_deg = (30, 45, 60)

angles_rad = tuple([deg_to_rad(angle) for angle in angles_deg])
ics = tuple([np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)]) for theta in angles_rad])

# Part A
# Varying vs Constant Gravity
sims_varying = tuple([ODE(rhs_varying_g, ic, ti=0, dt=0.01, tf=400,terminate=terminate) for ic in ics])
for sim in sims_varying:
    sim.run()
sims_constant = tuple([ODE(rhs_constant_g, ic, ti=0, dt=0.01, tf=400,terminate=terminate) for ic in ics])
for sim in sims_constant:
    sim.run()

minx0 = min(np.size(sims_varying[0].X_series[:,0]),np.size(sims_constant[0].X_series[:,0]))
minx1 = min(np.size(sims_varying[1].X_series[:,0]),np.size(sims_constant[1].X_series[:,0]))
minx2 = min(np.size(sims_varying[2].X_series[:,0]),np.size(sims_constant[2].X_series[:,0]))
miny0 = min(np.size(sims_varying[0].X_series[:,1]),np.size(sims_constant[0].X_series[:,1]))
miny1 = min(np.size(sims_varying[1].X_series[:,1]),np.size(sims_constant[1].X_series[:,1]))
miny2 = min(np.size(sims_varying[2].X_series[:,1]),np.size(sims_constant[2].X_series[:,1]))
min0 = min(minx0,miny0)
min1 = min(minx1,miny1)
min2 = min(minx2,miny2)

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot((sims_varying[0].X_series[:min0,0]-sims_constant[0].X_series[:min0,0])/1000,
        (sims_varying[0].X_series[:min0,1]-sims_constant[0].X_series[:min0,1])/1000,label=rf"$\theta = 30^{{\circ}}$")
ax.plot((sims_varying[1].X_series[:min1,0]-sims_constant[1].X_series[:min1,0])/1000,
        (sims_varying[1].X_series[:min1,1]-sims_constant[1].X_series[:min1,1])/1000, label=rf"$\theta = 45^{{\circ}}$")
ax.plot((sims_varying[2].X_series[:min2,0]-sims_constant[2].X_series[:min2,0])/1000,
        (sims_varying[2].X_series[:min2,1]-sims_constant[2].X_series[:min2,1])/1000, label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8a")
plt.savefig("../../figures/Chapter2/Problem2_8a",dpi=300)

# Part B
fig, ax = plt.subplots(1,1)

ax.plot(sims_varying[0].t_series[:minx0],(sims_varying[0].X_series[:minx0,0]-sims_constant[0].X_series[:minx0,0])/1000,
        label=rf"$\theta = 30^{{\circ}}$")
ax.plot(sims_varying[1].t_series[:minx1],(sims_varying[1].X_series[:minx1,0]-sims_constant[1].X_series[:minx1,0])/1000,
        label=rf"$\theta = 45^{{\circ}}$")
ax.plot(sims_varying[2].t_series[:minx2],(sims_varying[2].X_series[:minx2,0]-sims_constant[2].X_series[:minx2,0])/1000,
        label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8b")
plt.savefig("../../figures/Chapter2/Problem2_8b",dpi=300)

# Part C
fig, ax = plt.subplots(1,1)

ax.plot(sims_varying[0].t_series[:miny0],(sims_varying[0].X_series[:miny0,1]-sims_constant[0].X_series[:miny0,1])/1000,
        label=rf"$\theta = 30^{{\circ}}$")
ax.plot(sims_varying[1].t_series[:miny1],(sims_varying[1].X_series[:miny1,1]-sims_constant[1].X_series[:miny1,1])/1000,
        label=rf"$\theta = 45^{{\circ}}$")
ax.plot(sims_varying[2].t_series[:miny2],(sims_varying[2].X_series[:miny2,1]-sims_constant[2].X_series[:miny2,1])/1000,
        label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("y [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8c")
plt.savefig("../../figures/Chapter2/Problem2_8c",dpi=300)
