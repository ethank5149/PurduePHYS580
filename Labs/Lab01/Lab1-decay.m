% Radioactive decay by Kevin Berwick (slight changes by Denes Molnar)
% based on 'Computational Physics' book by N Giordano and H Nakanishi
% Section 1.2 p2
%
% Solves dN/dt = -N/tau, with initial condition N(t=0) = N0
%
clear;
%
% Set parameters of the calculation. Inputs could also be read from terminal via 'input'
%
% initial condition
N0 = 1000; 
% mean lifetime, measured in units of mean life time => tau = 1
tau = 1;
% time step, in units of tau
dt = 0.05;
% number of time steps to calculate
nsteps = 100;
%
% Initialize N_i and t_i as vectors of dimension nsteps+1, with values being all zeros
N_nuclei = zeros(nsteps + 1, 1);
time = zeros(nsteps + 1, 1);
% set initial conditions by assigning the values of N(1), t(1)
N_nuclei(1) = N0;
time(1) = 0;
%
% Loop over timesteps and calculate the numerical solution via Euler approximation
for i = 1:nsteps
	N_nuclei(i+1) = N_nuclei(i) - N_nuclei(i)/tau*dt;
	time(i+1) = time(i) + dt;
end
%
% For comparison, calculate exact solution as a vector, in one shot
tvalues = 0:0.02:5;
N_exact = N0 * exp(-tvalues/tau);
%
% Plot both Euler and exact solutions
plot(time, N_nuclei, 'r+', tvalues, N_exact, 'b', 'MarkerSize', 2);
xlabel('Time in units of mean lifetime')
ylabel('Number of undecayed nuclei')
title('Euler approximation')

%EOF