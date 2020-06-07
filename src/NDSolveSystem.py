###############################################################################
#     ========     |  NDSolveSystem                                           #
#     \\           |  A basic ODE class for solving DE systems                #
#      \\          |                                                          #
#      //          |  Author: Ethan Knox                                      #
#     //           |  Website: https://www.github.com/ethank5149              #
#     ========     |  MIT License                                             #
###############################################################################
###############################################################################
# License                                                                     #
# Copyright 2020 Ethan Knox                                                   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
###############################################################################

import numpy as np


class ODE():
    def __init__(self, rhs, X0, dt=0.01, ti=0, tf=1, terminate=lambda *args: False, method='euler', eps=0.01):
        self.X = X0  # Initial State
        # Initial Time, Initial Timestep,Final Time
        self.t, self.dt, self.tf = ti, dt, tf
        self.t_series = []  # t Data
        self.X_series = []  # X Data
        self.eps = eps  # Used in rkf45
        self.rhs = rhs  # ODE System
        self.terminate = terminate  # Termination Condition
        self.method = method  # Solver method

        if self.method == 'euler':
            self.update = self.euler_step
        elif self.method == 'heun':
            self.update = self.heun_step
        elif self.method == 'rk4':
            self.update = self.rk4_step
        elif self.method == 'rkf45':
            self.update = self.rkf45_step
        else:
            self.update = self.euler_step

    def euler_step(self):
        # Forward Euler
        self.X = self.X + self.dt * self.rhs(self.t, self.X)
        self.t += self.dt

    def heun_step(self):
        # RK2/Heun
        y_tilde = self.X+self.dt*self.rhs(self.t, self.X)
        self.X = self.X + 0.5*self.dt * \
            (self.rhs(self.t, self.X)+self.rhs(self.t+self.dt, self.X + y_tilde))
        self.t += self.dt

    def rk4_step(self):
        # # RK4
        k1 = self.rhs(self.t, self.X)
        k2 = self.rhs(self.t+0.5*self.dt, self.X + 0.5*self.dt*k1)
        k3 = self.rhs(self.t+0.5*self.dt, self.X + 0.5*self.dt*k2)
        k4 = self.rhs(self.t+self.dt, self.X + self.dt*k3)
        self.X = self.X + self.dt * (k1+2*k2+2*k3+k4)/6
        self.t += self.dt

    def rkf45_step(self):
        # RKF45
        k1 = self.rhs(self.t,                 self.X)
        k2 = self.rhs(self.t + (1/4)*self.dt, self.X + (1/4)*self.dt*k1)
        k3 = self.rhs(self.t + (3/8)*self.dt, self.X +
                      (3/8)*self.dt*((1/4)*k1 + (3/4)*k2))
        k4 = self.rhs(self.t+(12/13)*self.dt, self.X + (12/13) *
                      self.dt*((161/169)*k1 - (600/169)*k2 + (608/169)*k3))
        k5 = self.rhs(self.t + self.dt, self.X + self.dt*((8341/4104)
                                                          * k1 - (32832/4104)*k2 + (29440/4104)*k3 - (845/4104)*k4))
        k6 = self.rhs(self.t + (1/2)*self.dt, self.X + (1/2)*self.dt*((-6080/10260)
                                                                      * k1+(41040/10260)*k2-(28352/10260)*k3+(9295/10260)*k4-(5643/10260)*k5))

        yt = self.X + self.dt * (2375*k1 + 11264*k3 + 10985*k4 - 4104*k5)/20520
        zt = self.X + self.dt * \
            (33440*k1+146432*k3 + 142805*k4-50787*k5+10260*k6)/282150

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
        np.savetxt(filename, output_array,
                   delimiter=',', header=','.join(header))
