%
% RK2 calculation for the motion of a non-linear pendulum
%
% by Kevin Berwick, edited slightly by H. Nakanishi and D. Molnar
% based on 'Computational Physics' book by N Giordano and H Nakanishi,
% Section 3.1
%
clear all;
%
% Initialize physical parameters
% - ordinary pendulum
length = input('Length of pendulum [m]: ');
theta0 = input('Initial angle [rad]: ');
omega0 = input('initial angular velocity [rad/s]: ');
% - dissipation and driving force
q      = input('dissipation coefficient [1/s]: ');
fD     = input('driving coefficient [1/s^2]: ');
OmegaD = input('driving angular frequency [rad/s]: ');
%
g = 9.8;
Omega0sq = g / length;  % natural ang. frequency squared
%
% Initialize computational parameters
%
dt     = input('dt [s]: ');
nsteps = input('number of steps: ');
%
% Initialize arrays for time, angle, angular velocity
% and set initial conditions
%
time   = zeros(nsteps+1,1);
theta  = zeros(nsteps+1,1);
omega  = zeros(nsteps+1,1);
% we could track E, but it can be reconstructed at end from theta & omega
%energy = zeros(nsteps+1,1);
%
theta(1) = theta0;
omega(1) = omega0;
%energy(1) = 0.5 * length^2 * omega0^2 + g * length * (1 - cos(theta0));
%
% Loop over the timesteps using the RK2 approximaton
%
for i = 1:nsteps
  % RK2 half step
	omp = omega(i) + 0.5 * dt * ( - Omega0sq * sin(theta(i)) - q * omega(i) ...
		                            + fD * sin(OmegaD*time(i)) );
  thp = theta(i) + 0.5 * dt * omega(i);
	tp  = time(i)  + 0.5 * dt;
  % RK2 full step
	omega(i+1) = omega(i) + dt * ( - Omega0sq * sin(thp) - q * omp ...
                                 + fD * sin(OmegaD*tp) );
  th1 = theta(i) + omp * dt;
  % cut theta modulo 2pi, map to [-pi,pi] range, store constrained value
  % -> floor(x) returns highest integer that is <= x
  th1 = th1 - floor(th1 / (2*pi)) * 2*pi;
  if (th1 > pi)
    th1 = th1 - 2*pi;
  end;
  theta(i+1) = th1;
  %energy(step+1) = 0.5*length^2*omega(step+1)^2+g*length*(1-cos(theta(step+1)));
	time(i+1) = time(i) + dt;
end;
%
% Show results as a 2x2 array of plots
%
% reconstruct energy
energy = 0.5 * length^2 * omega.^2 + g * length * (1 - cos(theta));
% plots
subplot(2, 2, 1);
plot(time, theta, 'r', 'LineWidth', 3);
legend('Angle');
xlabel('time [s]');
ylabel('theta [rad]');
%
subplot(2, 2, 2);
plot(time, omega, 'b', 'LineWidth', 3);
legend('Angular velocity');
xlabel('time [s]');
ylabel('omega [rad/s]');
%
subplot(2, 2, 3);
plot(theta, omega, 'm', 'LineWidth', 3);
%plot(omega, energy, 'm', 'LineWidth', 3);
legend('Phase portrait');
xlabel('theta [rad]');
ylabel('omega [rad/s]');
title( ['RK2: dt = ', num2str(dt), ', th(0) = ', num2str(theta0), ...
        ', om(0) = ', num2str(omega0), ', l = ', num2str(length)] )
%
subplot(2, 2, 4);
plot(time, energy, 'g', 'LineWidth', 3);
legend('Energy');
%ylim([0,2*energy(1)]);
xlabel('time [s]');
ylabel('energy/mass [J/kg]');
title([', q = ', num2str(q), ', f_D = ', num2str(fD), ...
       ', OmegaD = ', num2str(OmegaD)])
