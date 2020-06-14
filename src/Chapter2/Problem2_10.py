########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 10                                                                           #
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
import threading
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
    try:
        return air.density*(1-a*y/T_0)**alpha
    except RuntimeWarning:
        return air.density


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


def plot_terrain(ax,elev,dist):
    if elev < 0:
        ground = mpatches.Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), (dist/1000 - ax.get_xlim()[0]),
                                    (0 - ax.get_ylim()[0]), color='green')
        valley = mpatches.Rectangle((dist/1000, ax.get_ylim()[0]), (ax.get_xlim()[1] - dist/1000),
                                    (elev/1000-ax.get_ylim()[0]), color='green')
        ax.add_patch(ground)
        ax.add_patch(valley)
    else:
        ground = mpatches.Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), (ax.get_xlim()[1] - ax.get_xlim()[0]),
                                    (0 - ax.get_ylim()[0]), color='green')
        ax.add_patch(ground)
        cliff = mpatches.Rectangle((dist/1000, 0), (ax.get_xlim()[1] - dist/1000), elev/1000, color='green')
        ax.add_patch(cliff)



angles = (30, 35, 40, 45, 50, 55, 60)
distance_to_cliff = 10000
elevation_of_cliff = 10000
t = np.linspace(0, 400, 40000)

# # Part A
solns = [euler(rhs, f_ics(v0, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for angle in angles]

fig, ax = plt.subplots(1, 1)
for angle, soln in zip(angles,solns):
    ax.plot(soln[0]/1000, soln[1]/1000, label=rf"$\theta = {angle}^{{\circ}}$")

plot_terrain(ax,elevation_of_cliff,distance_to_cliff)

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

plot_terrain(ax,elevation_of_cliff,distance_to_cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10b")
plt.savefig("../../figures/Chapter2/Problem2_10b", dpi=300)


# Part C
# Fixing the launch angle to 60 degrees
angle = 60

elevation_of_cliff = -10000
velocities = np.linspace(100,1000,25)
solns = [euler(rhs, f_ics(v, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for v in velocities]

fig, ax = plt.subplots(1, 1)
solns = [euler(rhs, f_ics(v, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for v in velocities]
for v, soln in zip(velocities,solns):
    ax.plot(soln[0]/1000, soln[1]/1000)
plot_terrain(ax,elevation_of_cliff,distance_to_cliff)

ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("'Shooting' at an Decline")
plt.suptitle("Problem 2.10c")
plt.subplots_adjust(wspace=0.4)
plt.savefig("../../figures/Chapter2/Problem2_10c", dpi=300)


# Part D
elevation_of_cliff = 10000
fig, ax = plt.subplots(1, 1)
solns = [euler(rhs, f_ics(v, angle), t, terminate=partial(terminate,dist=distance_to_cliff,
                                                       elev=elevation_of_cliff)) for v in velocities]
for v, soln in zip(velocities,solns):
    ax.plot(soln[0]/1000, soln[1]/1000)
plot_terrain(ax,elevation_of_cliff,distance_to_cliff)

ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("'Shooting' at a Incline")
plt.suptitle("Problem 2.10d")
plt.subplots_adjust(wspace=0.4)
plt.savefig("../../figures/Chapter2/Problem2_10d", dpi=300)


# Part E
# We then solve the resulting boundary value problem to determine the minimum velocity for a given cliff height
def shoot_const_angle(elev,theta,vel_bound):
    def f_comp(vel):
        soln = euler(rhs, f_ics(vel, theta), t, terminate=partial(terminate, dist=distance_to_cliff, elev=elev))
        return np.sqrt((soln[0, -1] - distance_to_cliff) ** 2 + (soln[1, -1] - elev) ** 2)

    optimum_velocity = minimize_scalar(f_comp,bounds=vel_bound).x
    return optimum_velocity

elevations = np.linspace(-1000,10000,100)

# class RunForCertainElevation(threading.Thread):
#     angle = None
#     elevation = None
#     optimum_velocity = None
#
#     def __init__(self, certain_angle, certain_elevation):
#         super().__init__()
#         self.angle = certain_angle
#         self.elevation = certain_elevation
#
#     def run(self):
#         self.optimum_velocity = shoot_const_angle(self.elevation,self.angle,(200,2000))
#
#
# class RunForCertainAngle(threading.Thread):
#     angle = None
#     optimum_velocities = None
#
#     def __init__(self, angle):
#         super().__init__()
#         self.angle = angle
#
#     def run(self):
#         velocities = []
#         threads = dict()
#         for elev in elevations:
#             threads[str(elev)] = RunForCertainElevation(self.angle, elev)
#             threads[str(elev)].start()
#
#         for elev, thread in threads.items():
#             thread.join()
#
#         for elev in elevations:
#             velocities.append(threads[str(elev)].optimum_velocity)
#
#         self.optimum_velocities = np.asarray(velocities)
#
#
# # First Threading Implementation
# class RunForCertainAngle(threading.Thread):
#     angle = None
#     optimum_velocities = None
#
#     def __init__(self, certain_angle):
#         super().__init__()
#         self.angle = certain_angle
#
#     def run(self):
#         data = np.asarray([shoot_const_angle(elev,self.angle,(200,2000)) for elev in elevations])
#         self.optimum_velocities = data


angles = (45, 50, 60, 70, 80)
# threads = dict()
# for angle in angles:
#     threads[str(angle)] = RunForCertainAngle(angle)
#     print(f"Starting thread for angle {angle}")
#     threads[str(angle)].start()
#
# for angle, thread in threads.items():
#         thread.join()
#         print(f"Joining thread for angle {angle}")
#
# fig, ax = plt.subplots(1, 1)
# for angle in angles:
#     ax.plot(elevations / 1000, threads[str(angle)].optimum_velocities, label=rf"$\theta = {angle}^{{\circ}}$")


# No Threading
data = np.asarray([[shoot_const_angle(elev,angle,(200,2000)) for elev in elevations] for angle in angles])

fig, ax = plt.subplots(1, 1)
for angle,velocities in zip(angles,data):
    ax.plot(elevations/1000, velocities, label=rf"$\theta = {angle}^{{\circ}}$")
ax.grid()
ax.set_xlabel("Elevation [km]")
ax.set_ylabel("Minimum Velocity [m/s]")
ax.set_title(f"Minimum $v_0$ vs. Elevation for Various Firing Angles")
plt.legend()
plt.suptitle("Problem 2.10e")
plt.savefig("../../figures/Chapter2/Problem2_10e", dpi=300)