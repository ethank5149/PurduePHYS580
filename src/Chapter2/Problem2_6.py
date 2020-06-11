########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 6                                                                            #
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
# Including
from lib.NDSolveSystem import ODE
import numpy as np
from matplotlib import pyplot as plt

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X):
    return np.array([X[2], X[3], 0, -g])


def terminate(X):
    return X[1] < 0


def deg_to_rad(theta):
    return np.pi*theta/180


def x(t, x0, y0, dx0, dy0):
    return x0+dx0*t


def y(t, x0, y0, dx0, dy0):
    return y0+dy0*t-0.5*g*t**2


x0 = y0 = 0
v0 = 700
angles_deg = (30, 35, 40, 45, 50, 55)

angles_rad = tuple([deg_to_rad(angle) for angle in angles_deg])
ics = tuple([np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)])
             for theta in angles_rad])

sims = tuple([ODE(rhs, ic, ti=0, dt=0.01, tf=200, terminate=terminate)
              for ic in ics])
for sim in sims:
    sim.run()

xs = tuple([x(sim.t, *(sim.X_series[:, 0])) for sim in sims])
ys = tuple([y(sim.t, *(sim.X_series[:, 0])) for sim in sims])

# Plotting
# First Plot
fig, ax = plt.subplots(1, 1)
for angle, sim in zip(angles_deg, sims):
    ax.plot(sim.X_series[0]/1000, sim.X_series[1]/1000,
            label=rf"$\theta = {angle}^{{\circ}}$")
ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.6a")
plt.savefig("../../figures/Chapter2/Problem2_6a", dpi=300)

# Second Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for angle, sim, _x in zip(angles_deg, sims, xs):
    ax1.plot(sim.t, np.absolute(
        sim.X_series[0]-_x), label=rf"$\theta = {angle}^{{\circ}}$")
for angle, sim, _y in zip(angles_deg, sims, ys):
    ax2.plot(sim.t, np.absolute(
        sim.X_series[1]-_y), label=rf"$\theta = {angle}^{{\circ}}$")

ax1.legend()
ax1.grid()
ax1.set_ylabel("x [m]")
ax1.set_title("Absolute Error")

ax2.legend()
ax2.grid()
ax2.set_xlabel("t [s]")
ax2.set_ylabel("y [m]")

plt.suptitle("Problem 2.6b")
plt.savefig("../../figures/Chapter2/Problem2_6b", dpi=300)
