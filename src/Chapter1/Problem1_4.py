########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 4                                                                            #
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
from lib.NDSolveSystem import ODE, SymplecticODE
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X, ta,tb):
    return np.array([-X[0]/ta,X[0]/ta-X[1]/tb])


def Na(t,ta,Na0):
    return Na0*np.exp(-t/ta)


def Nb(t,ta,tb,Na0,Nb0):
    return Na0*(tb/(ta-tb))*np.exp(-(ta+tb)*t/(ta*tb))*(np.exp(t/tb)-np.exp(t/ta))+Nb0*np.exp(-t/tb)


ta1,ta2,ta3 = 1,3,4
tb1,tb2,tb3 = 4,2,1
Na0,Nb0 = 1,0

sim1 = ODE(partial(rhs, ta=ta1,tb=tb1), np.array([Na0,Nb0]), ti=0, dt=0.1, tf=10)
sim2 = ODE(partial(rhs, ta=ta2,tb=tb2), np.array([Na0,Nb0]), ti=0, dt=0.1, tf=10)
sim3 = ODE(partial(rhs, ta=ta3,tb=tb3), np.array([Na0,Nb0]), ti=0, dt=0.1, tf=10)

sim1.run()
sim2.run()
sim3.run()

# Plotting
fig,axs = plt.subplots(2,3)

axs[0,0].plot(sim1.t_series, sim1.X_series[0], label=r"$N_A$")
axs[0,0].plot(sim1.t_series, sim1.X_series[1], label=r"$N_B$")

axs[0,1].plot(sim2.t_series, sim2.X_series[0], label=r"$N_A$")
axs[0,1].plot(sim2.t_series, sim2.X_series[1], label=r"$N_B$")

axs[0,2].plot(sim3.t_series, sim3.X_series[0], label=r"$N_A$")
axs[0,2].plot(sim3.t_series, sim3.X_series[1], label=r"$N_B$")

axs[1,0].plot(sim1.t_series, sim1.X_series[0]-Na(sim1.t_series,ta1,Na0), label=r"$N_A$")
axs[1,0].plot(sim1.t_series, sim1.X_series[1]-Nb(sim1.t_series,ta1,tb1,Na0,Nb0), label=r"$N_B$")
axs[1,1].plot(sim2.t_series, sim2.X_series[0]-Na(sim2.t_series,ta2,Na0), label=r"$N_A$")
axs[1,1].plot(sim2.t_series, sim2.X_series[1]-Nb(sim2.t_series,ta2,tb2,Na0,Nb0), label=r"$N_B$")
axs[1,2].plot(sim3.t_series, sim3.X_series[0]-Na(sim3.t_series,ta3,Na0), label=r"$N_A$")
axs[1,2].plot(sim3.t_series, sim3.X_series[1]-Nb(sim3.t_series,ta3,tb3,Na0,Nb0), label=r"$N_B$")

for i in range(2):
    for j in range(3):
        axs[i,j].legend()
        axs[i,j].grid()
        axs[i,j].set_xlabel("t")
        axs[0,j].set_ylabel(r"$N$")
        axs[1,j].set_ylabel(r"Error")

axs[0,0].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta1}}}{{{tb1}}}$")
axs[0,1].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta2}}}{{{tb2}}}$")
axs[0,2].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta3}}}{{{tb3}}}$")
plt.suptitle("Problem 1.4")
for ax in axs.flat:
    ax.label_outer()
plt.subplots_adjust(hspace=0.2,wspace=0.3)

plt.savefig("../../figures/Chapter1/Problem1_4",dpi=300)
