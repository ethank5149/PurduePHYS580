%
% Simulation of chaotic tumbing of Hyperion, Saturn's moon, with Euler Cromer method
% Based on 'Computational Physics' book by N Giordano and H Nakanishi, Section 4.6
%
% by Kevin Berwick, revised to plot in place by Hisao Nakanishi, modified by Denes Molnar
%
clear all;
%
% Read initial conditions
% 
% position and velocity
x0 = input('Hyperion CM location: x [HU]: ');
y0 = input('                      y [HU]: ');
vx = input('                 v_x [HU/Hyr]: ');
vy = input('                 v_y [HU/Hyr]: ');
% angle (-pi,pi] and angular velocity
theta0 = input('Axis angle [rad]: ');
omega0 = input('Axis angular velocity [rad/Hyr]: ');
%
% Initialize computational parameters
%
t_end = input('simulation end time [Hyr]: ');
dt = input('dt [Hyr]: ');
nsteps = floor(t_end / dt + 0.5);
% storage for time history
x = zeros(nsteps + 1, 1);
y = zeros(nsteps + 1, 1);
theta = zeros(nsteps + 1, 1);
omega = zeros(nsteps + 1, 1);
t = zeros(nsteps + 1, 1);
% initialize using initconds
x(1) = x0;
y(1) = y0;
theta(1) = theta0;
omega(1) = omega0;
t(1) = 0;
%
% Euler-Cromer time evolution
%
GM = 4 * pi^2;  % G * M_saturn in HU-Hyr units
for i = 1:nsteps
    % radius for CM
    r = sqrt(x(i)^2 + y(i)^2);
    % update CM position
    vx_new = vx - GM * x(i) * dt / r^3;
    vy_new = vy - GM * y(i) * dt / r^3;
    x(i+1) = x(i) + vx_new * dt;
    y(i+1) = y(i) + vy_new * dt;
    t(i+1) = t(i) + dt;
    % update axis tilt and angular velocity
    omega(i+1) = omega(i) - dt * 3. * GM / r^5 ...
                 * (x(i) * sin(theta(i)) - y(i) * cos(theta(i))) ...
                 * (x(i) * cos(theta(i)) + y(i) * sin(theta(i)));
    theta(i+1) = theta(i) + omega(i+1) * dt;
    % map angle to (-pi,pi]
    x1 = theta(i+1) - floor(theta(i+1)/(2*pi)) * 2*pi;
    if (x1 > pi)
        theta(i+1) = x1 - 2*pi;
    end
    % update velocity (not stored in time history)
    vx = vx_new;
    vy = vy_new;
end
%
% Plot the results
%
clf;
subplot(2, 2, 1);
plot(x, y, 'r');
xlabel('x [HU]');
ylabel('y [HU]');
title('CM Position');
%
subplot(2, 2, 2);
plot(theta, omega, 'g');
xlabel('Angle [rad]');
ylabel('Angular velocity [rad/Hyr]');
title('Axis phase portrait');
%
subplot(2, 2, 3);
plot(t, theta, 'r');
xlabel('time [Hyr]');
ylabel('Angle [rad]');
title('Axis orientation');
%
subplot(2, 2, 4);
plot(t, omega, 'b');
xlabel('time [Hyr]');
ylabel('Angular velocity [rad/Hyr]');
title('Axis angular velocity');