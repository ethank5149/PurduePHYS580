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


import numpy as np
from matplotlib import pyplot as plt
from lib.DSolve import euler


# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X):
    return np.array([-g, ])


# Define an initial condition
x0_0 = 100
t1 = np.linspace(0,10,20)  # dt = 0.5
t2 = np.linspace(0,10,40)  # dt = 0.25
t3 = np.linspace(0,10,50)  # = dt = 0.2

y1 = euler(rhs,[x0_0,],t1)[0]
y2 = euler(rhs,[x0_0,],t2)[0]
y3 = euler(rhs,[x0_0,],t3)[0]

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t1, y1, label=f"dt = 0.5")
ax1.plot(t2, y2, label=f"dt = 0.25")
ax1.plot(t3, y3, label=f"dt = 0.2")

ax2.plot(t1, 100*(y1 - (x0_0 - g*t1))/(x0_0 - g*t1), label=f"dt = 0.5")
ax2.plot(t2, 100*(y2 - (x0_0 - g*t2))/(x0_0 - g*t2), label=f"dt = 0.25")
ax2.plot(t3, 100*(y3 - (x0_0 - g*t3))/(x0_0 - g*t3), label=f"dt = 0.2")

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

plt.suptitle("Problem 1.1")
plt.subplots_adjust(hspace=0.3, top=0.85)
plt.savefig("../../figures/Chapter1/Problem1_1", dpi=300)
