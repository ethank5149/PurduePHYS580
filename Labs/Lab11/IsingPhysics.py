import numpy as np
import pandas as pd
from tqdm import trange
from numba import jit, prange
import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def isingmodel_driver(
    nrows, # Number of rows
    ncols, # Number of columns
    h,    # Magnetic Field Strength
    J,    # Ferromagnetic Coupling Constant
    kB,   # Boltzmann Constant
    nR,   # Number of relaxation iterations done for each temperature
    sS,   # Number of observations from each temperature
    T   # Initial Temperature
    ):

    scale = 1.0 / (nrows * ncols)  # Intrinsic properties only
    energy_S, magnetization_S = np.zeros(sS), np.zeros(sS)
    
    for s_idx in prange(sS):  # For each lattice in the sample
        grid = np.random.choice(np.array([1, -1]), size=(nrows,ncols))  # Reset the lattice
        for r_idx in range(nR):  # Relax the lattice
            for i in range(nrows):  # Iterate through the rows
                for j in range(ncols):  # Iterate through the columns
                    neighbors = grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1]
                    dE = 2.0 * (J * neighbors - h) * grid[i, j]
                    probability = 1.0 if dE < 0.0 else np.exp(-dE / (kB * T))
                    if np.random.random() < probability:
                        grid[i, j] = -grid[i, j]  # Flip!
        neighbors = np.array([[
            grid[_i + 1, _j] + grid[_i - 1, _j] + grid[_i, _j + 1] + grid[_i, _j - 1] 
            for _j in range(ncols)] for _i in range(nrows)])
        dE, dM = 2.0 * (J * neighbors - h) * grid, grid
        energy_S[s_idx] = -0.5 * np.sum(dE)
        magnetization_S[s_idx] = np.sum(dM)
    energy_T = scale * np.mean(energy_S)
    magnetization_T = scale * np.var(energy_S) * T ** (-2)
    c_T[t_idx] = scale * np.mean(magnetization_S)
    chi_T[t_idx] = scale * np.var(magnetization_S) * T ** (-1)
    return Ts, energy_T, c_T, magnetization_T, chi_T


@jit(nopython=True, parallel=True)
def isingmodel(
    nrows=25, # Number of rows
    ncols=25, # Number of columns
    h=0.0,    # Magnetic Field Strength
    J=1.0,    # Ferromagnetic Coupling Constant
    kB=1.0,   # Boltzmann Constant
    nR=250,   # Number of relaxation iterations done for each temperature
    sS=250,   # Number of observations from each temperature
    Ti=0.5,   # Initial Temperature
    Tf=3.0,   # Final Temperature
    nT=250    # Number of temperature values
    ):

    scale = 1.0 / (nrows * ncols)  # Intrinsic properties only
    Ts = np.linspace(Ti, Tf, nT)
    energy_S, magnetization_S = np.zeros(sS), np.zeros(sS)
    energy_T, magnetization_T, c_T, chi_T = np.zeros(nT), np.zeros(nT), np.zeros(nT), np.zeros(nT)

    for t_idx in prange(nT):
        T = Ts[t_idx]
        for s_idx in prange(sS):  # For each lattice in the sample
            grid = np.random.choice(np.array([1, -1]), size=(nrows,ncols))  # Reset the lattice
            for r_idx in range(nR):  # Relax the lattice
                for i in range(nrows):  # Iterate through the rows
                    for j in range(ncols):  # Iterate through the columns
                        neighbors = grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1]
                        dE = 2.0 * (J * neighbors - h) * grid[i, j]
                        probability = 1.0 if dE < 0.0 else np.exp(-dE / (kB * T))
                        if np.random.random() < probability:
                            grid[i, j] = -grid[i, j]  # Flip!
            neighbors = np.array([[
                grid[_i + 1, _j] + grid[_i - 1, _j] + grid[_i, _j + 1] + grid[_i, _j - 1] 
                for _j in range(ncols)] for _i in range(nrows)])
            dE, dM = 2.0 * (J * neighbors - h) * grid, grid
            energy_S[s_idx] = -0.5 * np.sum(dE)
            magnetization_S[s_idx] = np.sum(dM)
        energy_T[t_idx] = scale * np.mean(energy_S)
        magnetization_T[t_idx] = scale * np.var(energy_S) * T ** (-2)
        c_T[t_idx] = scale * np.mean(magnetization_S)
        chi_T[t_idx] = scale * np.var(magnetization_S) * T ** (-1)
    return Ts, energy_T, c_T, magnetization_T, chi_T


def main():
    Ts, energy_T, c_T, magnetization_T, chi_T = isingmodel()
    res = pd.DataFrame(
        np.vstack((Ts, energy_T, c_T, magnetization_T, chi_T)).T, 
        columns=['Temperature', 'Energy', 'Specific Heat Capacity', 'Magnetization', 'Magnetic Susceptibility'])
    res.to_csv('results.csv')
    
    fig, ax = plt.subplots(2, 2, figsize=(32, 32), dpi=200, constrained_layout=True)
    fig.suptitle('Ising Model (Metropolis-Hastings Algorithm)')
    ax[0,0].scatter(res['Temperature'], res[                 'Energy'], marker=',')
    ax[0,1].scatter(res['Temperature'], res[ 'Specific Heat Capacity'], marker=',')
    ax[1,0].scatter(res['Temperature'], res[          'Magnetization'], marker=',')
    ax[1,1].scatter(res['Temperature'], res['Magnetic Susceptibility'], marker=',')
    ax[0,0].set_ylabel('Energy')
    ax[0,1].set_ylabel('Magnetization')
    ax[1,0].set_ylabel('Specific Heat Capacity')
    ax[1,1].set_ylabel('Magnetic Susceptibility')
    ax[0,0].set_xlabel('Temperature')
    ax[0,1].set_xlabel('Temperature')
    ax[1,0].set_xlabel('Temperature')
    ax[1,1].set_xlabel('Temperature')
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    fig.savefig('results')


if __name__ == '__main__':
    main()