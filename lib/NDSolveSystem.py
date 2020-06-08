########################################################################################################################
#     ========     |  NDSolveSystem                                                                                    #
#     \\           |  A basic ODE class for solving DE systems                                                         #
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

# Symplectic Integrator Coefficients
# 1st Order
c_1_1 = 1
d_1_1 = 1
# 2nd Order
c_2_1, c_2_2 = 0, 1
d_2_1, d_2_2 = 0.5, 0.5
# Third Order
c_3_1, c_3_2, c_3_3 = 1, -2/3, 2/3
d_3_1, d_3_2, d_3_3 = -1/24, 3/4, 7/24
# Fourth Order
c_4_1 = c_4_4 = 1/(2*(2-2**(1/3)))
c_4_2 = c_4_3 = (1-2**(1/3))/(2*(2-2**(1/3)))
d_4_1 = d_4_3 = 1/(2-2**(1/3))
d_4_2, d_4_4 = -2**(1/3)/(2-2**(1/3)), 0

class ODE:
    def __init__(self, rhs, X0, dt=0.01, ti=0, tf=1, terminate=lambda *args: False, method='euler', eps=0.01):
        self.X = X0  # Initial State
        self.t, self.dt, self.tf = ti, dt, tf  # Initial, delta, and final Time
        self.t_series, self.X_series = [], []  # t,X Data
        self.eps = eps  # Used in rkf45
        self.rhs = rhs  # ODE System
        self.terminate = terminate  # Termination Condition
        self.method = method  # Solver method

        if self.method == 'euler':
            self.update = self.euler_step
        elif self.method == 'eulercromer':
            self.update = self.eulercromer_step
        elif self.method == 'heun':
            self.update = self.heun_step
        elif self.method == 'rk4':
            self.update = self.rk4_step
        elif self.method == 'rkf45':
            self.update = self.rkf45_step
        else:
            self.update = self.euler_step

    def euler_step(self):
        self.X = self.X + self.dt * self.rhs(self.t, self.X)
        self.t += self.dt

    def eulercromer_step(self):
        temp = self.X + self.dt * self.rhs(self.t, self.X)
        self.X = self.X + self.dt * self.rhs(self.t, temp)
        self.t += self.dt

    def heun_step(self):
        y_tilde = self.X + self.dt * self.rhs(self.t, self.X)
        self.X = self.X + 0.5 * self.dt * (self.rhs(self.t, self.X) + self.rhs(self.t + self.dt, self.X + y_tilde))
        self.t += self.dt

    def rk4_step(self):
        k1 = self.rhs(self.t, self.X)
        k2 = self.rhs(self.t + 0.5 * self.dt, self.X + 0.5 * self.dt * k1)
        k3 = self.rhs(self.t + 0.5 * self.dt, self.X + 0.5 * self.dt * k2)
        k4 = self.rhs(self.t + self.dt, self.X + self.dt * k3)
        self.X = self.X + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.t += self.dt

    def rkf45_step(self):
        k1 = self.rhs(self.t, self.X)
        k2 = self.rhs(self.t + 0.25 * self.dt, self.X + 0.25*self.dt*k1)
        k3 = self.rhs(self.t + (3/8) * self.dt, self.X + (3/8)*self.dt*(k1+3*k2)/4)
        k4 = self.rhs(self.t + (12/13) * self.dt, self.X + (12/13)*self.dt*(161*k1-600*k2+608*k3)/169)
        k5 = self.rhs(self.t + self.dt, self.X + self.dt*(8341*k1-32832*k2+29440*k3-845*k4)/4104)
        k6 = self.rhs(self.t + 0.5 * self.dt, self.X + 0.5*self.dt*(-6080*k1+41040*k2-28352*k3+9295*k4-5643*k5)/10260)

        yt = self.X + self.dt*(2375*k1+11264*k3+10985*k4-4104*k5)/20520
        zt = self.X + self.dt*(33440*k1+146432*k3+142805*k4-50787*k5+10260*k6)/282150

        s = (self.eps*self.dt/(2*np.linalg.norm(yt-zt)))**0.25
        if s < 0.75:
            self.dt = max((self.dt/2, 0.025))
        self.t += self.dt
        self.X = yt

    def run(self):
        while self.t < self.tf:
            self.X_series.append(list(self.X))
            self.t_series.append(self.t)
            self.update()
            if self.terminate(self.X):
                break
        self.X_series = np.array(self.X_series)
        self.t_series = np.array(self.t_series)

    def store(self, filename="output.dat"):
        header = ["t", ]+[f"x_{i}" for i in range(np.size(self.X))]
        output_array = np.hstack((np.array([self.t_series]).T, self.X_series))
        np.savetxt(filename, output_array, delimiter=',', header=','.join(header))


class SymplecticODE:
    def __init__(self, rhs, X0,dX0, dt=0.01, ti=0, tf=1, terminate=lambda *args: False, method='euler', eps=0.01):
        self.X,self.dX = X0,dX0  # Initial State
        self.t, self.dt, self.tf = ti, dt, tf  # Initial, delta, and final Time
        self.t_series, self.X_series,self.dX_series = [], [],[]  # t,X,dX Data
        self.eps = eps  # Used in rkf45
        self.rhs = rhs  # ODE System
        self.terminate = terminate  # Termination Condition
        self.method = method  # Solver method

        if self.method == 'eulercromer' or self.method == 'first_order':
            self.update = self.eulercromer_step
        elif self.method == 'verlet' or self.method == 'second_order':
            self.update = self.verlet_step
        elif self.method == 'third_order':
            self.update = self.third_order_step
        elif self.method == 'fourth_order':
            self.update = self.fourth_order_step
        else:
            self.update = self.eulercromer_step

    def eulercromer_step(self):
        self.dX = self.dX + d_1_1 * self.dt * self.rhs(self.t, self.X, self.dX)
        self.X = self.X + c_1_1 * self.dt * self.dX
        self.t += self.dt

    def verlet_step(self):
        dX1 = self.dX + d_2_1 * self.dt * self.rhs(self.t, self.X, self.dX)
        X1 = self.X + c_2_1 * self.dt * dX1
        self.dX = dX1 + d_2_2 * self.dt * self.rhs(self.t, X1, dX1)
        self.X = X1 + c_2_2 * self.dt * self.dX
        self.t += self.dt

    def third_order_step(self):
        dX1 = self.dX + d_3_1 * self.dt * self.rhs(self.t, self.X, self.dX)
        X1 = self.X + c_3_1 * self.dt * dX1
        dX2 = dX1 + d_3_2 * self.dt * self.rhs(self.t, X1, dX1)
        X2 = X1 + c_3_2 * self.dt * dX2
        self.dX = dX2 + d_3_3 * self.dt * self.rhs(self.t, X2, dX2)
        self.X = X2 + c_3_3 * self.dt * self.dX
        self.t += self.dt

    def fourth_order_step(self):
        dX1 = self.dX + d_4_1 * self.dt * self.rhs(self.t, self.X, self.dX)
        X1 = self.X + c_4_1 * self.dt * dX1
        dX2 = dX1 + d_4_2 * self.dt * self.rhs(self.t, X1, dX1)
        X2 = X1 + c_4_2 * self.dt * dX2
        dX3 = dX2 + d_4_3 * self.dt * self.rhs(self.t, X2, dX2)
        X3 = X2 + c_4_3 * self.dt * dX3
        self.dX = dX3 + d_4_4 * self.dt * self.rhs(self.t, X3, dX3)
        self.X = self.X3 + c_4_4 * self.dt * self.dX
        self.t += self.dt

    def run(self):
        while self.t < self.tf:
            self.dX_series.append(list(self.dX))
            self.X_series.append(list(self.X))
            self.t_series.append(self.t)
            self.update()
            if self.terminate(self.X):
                break
        self.dX_series = np.array(self.dX_series)
        self.X_series = np.array(self.X_series)
        self.t_series = np.array(self.t_series)

    def store(self, filename="output.dat"):
        header = ["t", ]+[f"x_{i}" for i in range(np.size(self.X))]+[f"dx_{i}" for i in range(np.size(self.dX))]
        output_array = np.hstack((np.array([self.t_series]).T, self.X_series, self.dX_series))
        np.savetxt(filename, output_array, delimiter=',', header=','.join(header))