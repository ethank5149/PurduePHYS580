########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 3                                                                            #
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
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
g = 9.81
l = 1
m = 1
Omega = np.sqrt(g/l)
T = 2*np.pi/Omega


def f(t, X):
    return np.array([X[1],-(g/l)*X[0]])




def E(t, X):
    return 0.5*m*l**2*X[1]**2+m*g*l*(1-np.cos(X[0]))


def diff_E(t, X):
    true_E = 0.5*m*l**2*X[1,0]**2+m*g*l*(1-np.cos(X[0,0]))
    return 100*(0.5*m*l**2*X[1]**2+m*g*l*(1-np.cos(X[0])) - true_E)/true_E


num_periods = 6
ic = [np.pi/3,0]
true_E = 0.5*m*l**2*ic[1]**2+m*g*l*(1-np.cos(ic[0]))
t1 = np.linspace(0,num_periods*T,num_periods*1000)
t2 = np.linspace(0,num_periods*T,num_periods*2000)
t3 = np.linspace(0,num_periods*T,num_periods*3000)
t4 = np.linspace(0,num_periods*T,num_periods*4000)
x1 = euler(f,ic,t1)
x2 = euler(f,ic,t2)
x3 = euler(f,ic,t3)
x4 = euler(f,ic,t4)
fig, ax = plt.subplots(1, 1)
ax.plot(t1, E(t1,x1), label=f"dt = {T/1000:0.4f}")
ax.plot(t2, E(t2,x2), label=f"dt = {T/2000:0.4f}")
ax.plot(t3, E(t3,x3), label=f"dt = {T/3000:0.4f}")
ax.plot(t4, E(t4,x4), label=f"dt = {T/4000:0.4f}")
for i in range(num_periods):
    plt.axvline(x=(i+1)*T, color='k', linestyle='--')
plt.axhline(y=true_E, color='purple')
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel("E [J]")
ax.set_title("Energy vs time Using Euler's Method")
plt.legend()
plt.suptitle("Problem 3.3")
plt.savefig("../../figures/Chapter3/Problem3_3", dpi=300)
