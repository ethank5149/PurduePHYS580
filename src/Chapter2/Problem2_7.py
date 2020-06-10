########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 7                                                                            #
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
from functools import partial
air = Constants.Air()

# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
a = 6.5e-3  # Adiabatic model parameter [K/m]
alpha = 2.5  # Adiabatic model parameter
y_0 = 1.0e4  # k_B*T/mg [m]
T_0 = 300  # Adiabatic model sea level temperature [K]
T_hot = 310.9278  # "Hot summer day" temperature ~ 100 [F]
T_cold = 266.4833  # "Cold winter day" temperature ~ 20 [F]


def rho_isothermal(y):
    return air.density*np.exp(-y/y_0)


def rho_adiabatic(y):
    return air.density*(1-a*y/T_0)**alpha


def B2(T):
    return B2_ref*(T_0/T)**alpha


def B2_eff(T,rho):
    return (rho/air.density)*B2(T)


def rhs_adiabatic(t, X, T):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(T,rho_adiabatic(X[1]))*v*X[2]/m, -g-B2_eff(T,rho_adiabatic(X[1]))*v*X[3]/m])


def rhs_isothermal(t, X, T):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(T,rho_isothermal(X[1]))*v*X[2]/m, -g-B2_eff(T,rho_isothermal(X[1]))*v*X[3]/m])


def terminate(X):
    return X[1] < 0


def deg_to_rad(theta):
    return np.pi*theta/180


x0 = y0 = 0
v0 = 700
angles_deg = (30, 45, 60)

angles_rad = tuple([deg_to_rad(angle) for angle in angles_deg])
ics = tuple([np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)]) for theta in angles_rad])

# Part A
# Adiabatic vs Isothermal, Normal Temperature
sims_adiabatic = tuple([ODE(partial(rhs_adiabatic, T=T_0), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_adiabatic:
    sim.run()
sims_isothermal = tuple([ODE(partial(rhs_isothermal, T=T_0), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_isothermal:
    sim.run()

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot(sims_isothermal[0].X_series[0]/1000, sims_isothermal[0].X_series[1]/1000,'r-',
        label=rf"Isothermal, $\theta = 30^{{\circ}}$")
ax.plot(sims_isothermal[1].X_series[0]/1000, sims_isothermal[1].X_series[1]/1000,'b-',
        label=rf"Isothermal, $\theta = 45^{{\circ}}$")
ax.plot(sims_isothermal[2].X_series[0]/1000, sims_isothermal[2].X_series[1]/1000,'k-',
        label=rf"Isothermal, $\theta = 60^{{\circ}}$")
ax.plot(sims_adiabatic[0].X_series[0]/1000, sims_adiabatic[0].X_series[1]/1000,'r--',
        label=rf"$Adiabatic, \theta = 30^{{\circ}}$")
ax.plot(sims_adiabatic[1].X_series[0]/1000, sims_adiabatic[1].X_series[1]/1000,'b--',
        label=rf"$Adiabatic, \theta = 45^{{\circ}}$")
ax.plot(sims_adiabatic[2].X_series[0]/1000, sims_adiabatic[2].X_series[1]/1000,'k--',
        label=rf"$Adiabatic, \theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Adiabatic vs. Isothermal Model")
plt.suptitle("Problem 2.7a")
plt.savefig("../../figures/Chapter2/Problem2_7a",dpi=300)

# Part B
# Adiabatic, Hot vs Cold
sims_cold = tuple([ODE(partial(rhs_adiabatic, T=T_cold), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_cold:
    sim.run()
sims_hot = tuple([ODE(partial(rhs_adiabatic, T=T_hot), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_hot:
    sim.run()

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot(sims_cold[0].X_series[0]/1000, sims_cold[0].X_series[1]/1000,'r-',
        label=rf"Cold, $\theta = 30^{{\circ}}$")
ax.plot(sims_cold[1].X_series[0]/1000, sims_cold[1].X_series[1]/1000,'b-',
        label=rf"Cold, $\theta = 45^{{\circ}}$")
ax.plot(sims_cold[2].X_series[0]/1000, sims_cold[2].X_series[1]/1000,'k-',
        label=rf"Cold, $\theta = 60^{{\circ}}$")
ax.plot(sims_hot[0].X_series[0]/1000, sims_hot[0].X_series[1]/1000,'r--',
        label=rf"$Hot, \theta = 30^{{\circ}}$")
ax.plot(sims_hot[1].X_series[0]/1000, sims_hot[1].X_series[1]/1000,'b--',
        label=rf"$Hot, \theta = 45^{{\circ}}$")
ax.plot(sims_hot[2].X_series[0]/1000, sims_hot[2].X_series[1]/1000,'k--',
        label=rf"$Hot, \theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Adiabatic Model, Hot vs. Cold Temperatures")
plt.suptitle("Problem 2.7b")
plt.savefig("../../figures/Chapter2/Problem2_7b",dpi=300)

import tqdm
# Part C
angles = np.linspace(np.pi / 6, np.pi / 3, 30)
temperatures = np.linspace(T_cold, T_hot, int(T_hot-T_cold))
max_ranges = []
for _,temperature in enumerate(tqdm.tqdm(temperatures,desc="Sweeping")):
    max_range = 0
    for angle in angles:
        y0 = [0, 0, v0 * np.cos(angle), v0 * np.sin(angle)]
        sim = ODE(partial(rhs_adiabatic,T=temperature),y0,ti=0,dt=0.01,tf=400,terminate=terminate)
        sim.run()
        max_range = max([max_range,sim.X_series[0,-1]])
    max_ranges.append(max_range/1000)

# Plotting
fig, ax = plt.subplots(1, 1)
ax.plot(temperatures-273.15, max_ranges)
ax.grid()
ax.set_xlabel("Temperature [C]")
ax.set_ylabel("Range [km]")
ax.set_title("Maximum Range Vs. Temperature")
plt.suptitle("Problem 2.7c")
plt.savefig("../../figures/Chapter2/Problem2_7c", dpi=300)

# ##############################
#
# # Account for the fact that the data can be different sizes due to the termination condition
# minx0 = min(np.size(adiabatic_x_0), np.size(isothermal_x_0))
# minx1 = min(np.size(adiabatic_x_1), np.size(isothermal_x_1))
# minx2 = min(np.size(adiabatic_x_2), np.size(isothermal_x_2))
# miny0 = min(np.size(adiabatic_y_0), np.size(isothermal_y_0))
# miny1 = min(np.size(adiabatic_y_1), np.size(isothermal_y_1))
# miny2 = min(np.size(adiabatic_y_2), np.size(isothermal_y_2))
# min0 = min(minx0, miny0)
# min1 = min(minx1, miny1)
# min2 = min(minx2, miny2)
#
# # Part A
# # Adiabatic vs Isothermal, Normal Temperature
#
#
# # Plotting
# fig, ax = plt.subplots(1,1)
# ax.plot(isothermal_x_0/1000, isothermal_y_0/1000,'r-', label=rf"Isothermal, $\theta = 30^{{\circ}}$")
# ax.plot(isothermal_x_1/1000, isothermal_y_1/1000,'b-', label=rf"Isothermal, $\theta = 45^{{\circ}}$")
# ax.plot(isothermal_x_2/1000, isothermal_y_2/1000,'k-', label=rf"Isothermal, $\theta = 60^{{\circ}}$")
# ax.plot(adiabatic_x_0/1000, adiabatic_y_0/1000,'r--', label=rf"$Adiabatic, \theta = 30^{{\circ}}$")
# ax.plot(adiabatic_x_1/1000, adiabatic_y_1/1000,'b--', label=rf"$Adiabatic, \theta = 45^{{\circ}}$")
# ax.plot(adiabatic_x_2/1000, adiabatic_y_2/1000,'k--', label=rf"$Adiabatic, \theta = 60^{{\circ}}$")
#
# ax.legend()
# ax.grid()
# ax.set_xlabel("x [km]")
# ax.set_ylabel("y [km]")
# ax.set_title("Adiabatic vs. Isothermal Model")
# plt.suptitle("Problem 2.7a")
# plt.savefig("../../figures/Chapter2/Problem2_7a",dpi=300)
#
# # Part B
#
#
# # Plotting
# fig, ax = plt.subplots(1,1)
# ax.plot(cold_x_0/1000, cold_y_0/1000,'r-', label=rf"Cold, $\theta = 30^{{\circ}}$")
# ax.plot(cold_x_1/1000, cold_y_1/1000,'b-', label=rf"Cold, $\theta = 45^{{\circ}}$")
# ax.plot(cold_x_2/1000, cold_y_2/1000,'k-', label=rf"Cold, $\theta = 60^{{\circ}}$")
# ax.plot(hot_x_0/1000, hot_y_0/1000,'r--', label=rf"Hot, $\theta = 30^{{\circ}}$")
# ax.plot(hot_x_1/1000, hot_y_1/1000,'b--', label=rf"Hot, $\theta = 45^{{\circ}}$")
# ax.plot(hot_x_2/1000, hot_y_2/1000,'k--', label=rf"Hot, $\theta = 60^{{\circ}}$")
#
# ax.legend()
# ax.grid()
# ax.set_xlabel("x [km]")
# ax.set_ylabel("y [km]")
# ax.set_title("Adiabatic Model, Hot vs. Cold Temperatures")
# plt.suptitle("Problem 2.7b")
# plt.savefig("../../figures/Chapter2/Problem2_7b",dpi=300)
#
# import tqdm
# # Part C
# angles = np.linspace(np.pi / 6, np.pi / 3, 30)
# temperatures = np.linspace(T_cold, T_hot, int(T_hot-T_cold))
# max_ranges = []
# for _,temperature in enumerate(tqdm.tqdm(temperatures,desc="Sweeping")):
#     max_range = 0
#     for angle in angles:
#         y0 = [0, 0, v0 * np.cos(angle), v0 * np.sin(angle)]
#         sim = ODE(partial(rhs_adiabatic,T=temperature),y0,ti=0,dt=0.01,tf=400,terminate=terminate)
#         sim.run()
#         max_range = max([max_range,sim.X_series[0][-1]])
#     max_ranges.append(max_range/1000)
#
# # Plotting
# fig, ax = plt.subplots(1, 1)
# ax.plot(temperatures-273.15, max_ranges)
# ax.grid()
# ax.set_xlabel("Temperature [C]")
# ax.set_ylabel("Range [km]")
# ax.set_title("Maximum Range Vs. Temperature")
# plt.suptitle("Problem 2.7c")
# plt.savefig("../../figures/Chapter2/Problem2_7c", dpi=300)
