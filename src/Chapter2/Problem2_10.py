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
from lib.DSolve import euler
from lib.Constants import Earth, Air
from lib.FindRoot import bisection, secant
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize_scalar
earth = Earth()
air = Air()


# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
a = 6.5e-3  # Adiabatic model parameter [K/m]
alpha = 2.5  # Adiabatic model parameter
y_0 = 1.0e4  # k_B*T/mg [m]
T_0 = 300  # Adiabatic model sea level temperature [K]
x0 = y0 = 0
v0 = 1000



def rho_adiabatic(y):
    return air.density*(1-a*y/T_0)**alpha


def B2_eff(rho):
    return (rho/air.density)*B2_ref


def rhs(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(rho_adiabatic(X[1]))*v*X[2]/m,
                     -earth.GM/(earth.radius+X[1])**2-B2_eff(rho_adiabatic(X[1]))*v*X[3]/m])


def terminate(X, dist, elev):
    return X[0]>0 and (X[1] < 0 and X[0]<dist) or (X[1] < elev and X[0]>dist)


def f_ics(v0, theta):
    return [x0, y0, v0*np.cos(np.pi*theta/180), v0*np.sin(np.pi*theta/180)]


angles = (30, 35, 40, 45, 50, 55, 60)
distance_to_cliff = 10000
elevation_of_cliff = 10000
t = np.linspace(0, 400, 40000)

# Part A
solns = [euler(rhs, f_ics(v0, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for angle in angles]

fig, ax = plt.subplots(1, 1)
for angle, soln in zip(angles,solns):
    ax.plot(soln[0]/1000, soln[1]/1000, label=rf"$\theta = {angle}^{{\circ}}$")

if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff/1000-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff/1000,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff/1000),
                                (elevation_of_cliff/1000-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff/1000, 0),(plt.xlim()[1]-distance_to_cliff/1000),
                               elevation_of_cliff/1000,color='green')
    ax.add_patch(cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10a")
plt.savefig("../../figures/Chapter2/Problem2_10a", dpi=300)


# Part B
elevation_of_cliff = -10000

solns = [euler(rhs, f_ics(v0, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for angle in angles]

fig, ax = plt.subplots(1, 1)
for angle, soln in zip(angles,solns):
    ax.plot(soln[0]/1000, soln[1]/1000, label=rf"$\theta = {angle}^{{\circ}}$")


if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff/1000-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff/1000,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff/1000),
                                (elevation_of_cliff/1000-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff/1000, 0),(plt.xlim()[1]-distance_to_cliff/1000),
                               elevation_of_cliff/1000,color='green')
    ax.add_patch(cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10b")
plt.savefig("../../figures/Chapter2/Problem2_10b", dpi=300)


# Part C
# Fixing the launch angle to 60 degrees
angle = 60
# We then solve the resulting boundary value problem to determine the minimum velocity for a given cliff height


def shoot(elev):
    def f_comp(vel):
        soln = euler(rhs, f_ics(vel, angle), t, terminate=partial(terminate, dist=distance_to_cliff, elev=elev))
        return np.sqrt((soln[0, -1] - distance_to_cliff) ** 2 + (soln[1, -1] - elev) ** 2)

    vel_opt = minimize_scalar(f_comp,bounds=(10,1000), method='bounded', tol=1.0).x
    return vel_opt


elevations = np.linspace(-10000,10000,100)
vels = [shoot(elevation) for elevation in elevations]

fig, ax = plt.subplots(1, 1)
plt.plot(elevations/1000,vels)
ax.grid()
ax.set_xlabel("Elevation [km]")
ax.set_ylabel("v [m/s]")
ax.set_title("Minimum Velocity vs. Target Elevation")
plt.suptitle("Problem 2.10c")
plt.savefig("../../figures/Chapter2/Problem2_10c", dpi=300)
