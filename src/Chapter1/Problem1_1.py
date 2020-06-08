########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 1                                                                            #
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


from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from lib.NDSolveSystem import ODE


# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


# Function Definitions
def rhs(t, X):
    return np.array([-g, ])


# Define an initial condition
x0_0 = 100

# Instantiate the simulations
sim1 = ODE(partial(rhs), np.array([x0_0, ]), ti=0, dt=0.3, tf=10, method='euler')
sim2 = ODE(partial(rhs), np.array([x0_0, ]), ti=0, dt=0.2, tf=10, method='euler')
sim3 = ODE(partial(rhs), np.array([x0_0, ]), ti=0, dt=0.1, tf=10, method='euler')

# Run the simulations
sim1.run()
sim2.run()
sim3.run()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(sim1.t_series, sim1.X_series[:, 0], label=f"dt = {sim1.dt}")
ax1.plot(sim2.t_series, sim2.X_series[:, 0], label=f"dt = {sim2.dt}")
ax1.plot(sim3.t_series, sim3.X_series[:, 0], label=f"dt = {sim3.dt}")

ax2.plot(sim1.t_series, 100 * (sim1.X_series[:, 0] - (x0_0 - g * sim1.t_series)) / (x0_0 - g * sim1.t_series),
         label=f"dt = {sim1.dt}")
ax2.plot(sim2.t_series, 100 * (sim2.X_series[:, 0] - (x0_0 - g * sim2.t_series)) / (x0_0 - g * sim2.t_series),
         label=f"dt = {sim2.dt}")
ax2.plot(sim3.t_series, 100 * (sim3.X_series[:, 0] - (x0_0 - g * sim3.t_series)) / (x0_0 - g * sim3.t_series),
         label=f"dt = {sim3.dt}")

ax1.set_title('y(t)')
ax1.legend()
ax1.grid()
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\frac{dy}{dt}$")

ax2.set_title('y(t) Percent Error')
ax2.legend()
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel(r"$\frac{dy}{dt}$")

plt.suptitle("Problem 1.1")  # \n"+r"$v(t)=v_0-gt$")
plt.subplots_adjust(hspace=0.3, top=0.85)
plt.savefig("../../figures/Chapter1/Problem1_1",dpi=300)
