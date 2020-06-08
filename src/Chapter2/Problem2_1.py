########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 1                                                                            #
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


def rhs(t, X, m,P):
    return np.array([P/(m*X[0]),])

def v(t,m,P,v0):
    return np.sqrt(2*P*t/m+v0**2)

P = 400
m = 70
v0 = 4

sim1 = ODE(partial(rhs, m=m,P=P), np.array([v0,]), ti=0, dt=0.5, tf=200)
sim2 = ODE(partial(rhs, m=m,P=P), np.array([v0,]), ti=0, dt=0.25, tf=200)
sim3 = ODE(partial(rhs, m=m,P=P), np.array([v0,]), ti=0, dt=0.125, tf=200)

sim1.run()
sim2.run()
sim3.run()

# Plotting
fig, ax = plt.subplots(2, 1,sharex=True)

ax[0].plot(sim1.t_series, sim1.X_series[:,0], label=f"dt = {sim1.dt}")
ax[0].plot(sim2.t_series, sim2.X_series[:,0], label=f"dt = {sim2.dt}")
ax[0].plot(sim3.t_series, sim3.X_series[:,0], label=f"dt = {sim3.dt}")
ax[1].plot(sim1.t_series, sim1.X_series[:,0]-v(sim1.t_series,m,P,v0), label=f"dt = {sim1.dt}")
ax[1].plot(sim2.t_series, sim2.X_series[:,0]-v(sim2.t_series,m,P,v0), label=f"dt = {sim2.dt}")
ax[1].plot(sim3.t_series, sim3.X_series[:,0]-v(sim3.t_series,m,P,v0), label=f"dt = {sim3.dt}")

ax[0].legend()
ax[0].grid()
ax[0].set_xlabel("t")
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel("t")

ax[0].set_ylabel(r"$v(t)$")
ax[1].set_ylabel(r"Error")

plt.suptitle("Problem 2.1")
plt.savefig("../../figures/Chapter2/Problem2_1",dpi=300)
