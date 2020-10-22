%
% Euler calculation of motion of simple pendulum
% by Kevin Berwick, edited slightly by H. Nakanishi and then by Denes Molnar
% based on 'Computational Physics' book by N. Giordano and H. Nakanishi,
% Section 3.1
%
% solves ODE system  (1)   dtheta/dt = omega
%                    (2)   domega/dt = -om0^2 * theta
% 
%   where om0^2 = (g/l)
%
clear all;
%
% Initialize physical parameters
%
length = input('length of pendulum [m]: ');
theta0 = input('initial angle wrt to vertical [rad]: ');
omega0 = input('initial angular velocity [rad/s]: ');
g = 9.8;  % SI units [m/s^2]
%
% Initialize computational parameters
%
dt = input('dt [s]: ');
nsteps = input('number of timesteps: ');
%
% Initialize arrays holding time, angle, and angular velocity
%
time = zeros(nsteps+1,1);
theta = zeros(nsteps+1,1);
omega = zeros(nsteps+1,1);
%
theta(1) = theta0;
omega(1) = omega0;
om0 = (g/length)^0.5;  % natural angular frequency
%
% Compute time evolution using the Euler approximaton
%
for i = 1:nsteps
	omega(i+1) = omega(i) - om0^2 * theta(i) * dt;
	theta(i+1) = theta(i) + omega(i)*dt;
	time(i+1)  = time(i) + dt;
end;
%
% Plot results for theta(t), omega(t), E(t) side by side
%
subplot(1, 3, 1);
plot(time, theta, 'r', 'LineWidth', 3);
legend('Angular displacement');
xlabel('time [s]');
ylabel('theta [rad]');
%
subplot(1, 3, 2);
plot(time, omega, 'b', 'LineWidth', 3);
legend('Angular velocity');
xlabel('time [s]');
ylabel('omega [rad/s]');
title(['Euler method with dt = ',num2str(dt), ', theta(0) = ', num2str(theta0),... 
      ', omega(0) = ', num2str(omega0), ', l = ', num2str(length)])
%
subplot(1, 3, 3);
plot(time, om0^2 * theta.^2 + omega.^2, 'g', 'LineWidth', 3);
legend('Energy');
xlabel('time [s]');
ylabel('normalized energy [1/s^2]');