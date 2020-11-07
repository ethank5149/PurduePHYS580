# radioactive decay code by Denes Molnar
#
# for PHYS 580, Lab 1, Fall 2020
#
# uses Euler's method to solve dN/dt = -N/tau, with initial condition N(t=0) = N0
#


from math import exp    # need exp()
import numpy as np      # numpy is not strictly necessary
                        # but for many calculations numpy arrays work faster than lists, so why not

import matplotlib       # for plotting
matplotlib.use('TkAgg') # this may be different or unnecessary on your system (how you get graphics)
import matplotlib.pyplot as plt 


# parameters
N0  = 1000   # initial number of atoms
tau = 1.     # mean lifetime - set to 1, i.e., measure time in units of tau
dt  = 0.05   # time step (in units of tau)
nsteps = 100 # number of time steps to take

# you can read keyboard input (keystrokes + ENTER) with input()
#nsteps = int( input("nsteps: ") )

# storage for number of atoms and time values, at each timestep 
Natoms = np.zeros(nsteps + 1)
times  = np.zeros(nsteps + 1)    # times could have been populated in one step, since dt = const

# initial conditions (Python indices start from 0)
Natoms[0] = N0
times[0] = 0.

# Euler update timestep by timestep
for i in range(nsteps):
   Natoms[i + 1] = Natoms[i] - Natoms[i] * dt / tau
   times[i + 1]  = times[i] + dt

# exact solution for comparison, in one shot
Nexact = np.array( [ N0 * exp(-t / tau)   for t in times ] )

# Plot both Euler and exact solutions
plt.xlabel('t / tau')    # time axis label
plt.ylabel('N(t)')       # number axis label
plt.title('Euler approximation') # plot title
plt.plot(times, Natoms, label = "Euler")   # Euler curve
plt.plot(times, Nexact, label = "exact")   # exact curve
plt.legend()             # create legends
plt.show()               # show plot


#EOF
