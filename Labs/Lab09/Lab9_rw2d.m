% Random walk on a 2 dimensional square lattice
%

clear all;

% input physical parameters
%
Nstep  = input('maximum number of steps: ');
Nwalks = input('number of walks: '); 

% default RNG is good enough
%
%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);


% initialize variables and arrays
%
x2ave = zeros(1, Nstep);
x     = zeros(2, Nstep + 1);
%
% generate each RW in its entirety
%
for i = 1:Nwalks
    % generate one random walk
    [x] = Lab9_generate_rw(Nstep);
    % plot the path (or skip it for speed, i.e., comment out the next 3 lines)
    clf;
    plot(x(1,1:end), x(2,1:end), "b", "LineWidth", 3);
    pause;
    % collect x^2 staistics
    for j = 1:Nstep
        x2ave(j) = x2ave(j) + x(1, j+1)^2 + x(2, j+1)^2;
    end
end
for j = 1:Nstep
    x2ave(j) = x2ave(j) / Nwalks;
end

%
% plot <r^2> and its standard deviation vs t data points in log-log plot
%

clf;

t = (1:Nstep);
logt = log10(t);
logr2 = log10(x2ave(1:Nstep));
scatter(logt, logr2, 15, 'r', 'x'); 

title('RW in d=2')
xlabel('log_{10} N');
ylabel('log_{10} <r^2>');

hold on;

% do a linear least-squares fit and plot it
%
disp("polynomial fit to log(r^2) vs log(t)");

p = polyfit(logt, logr2, 1)
logr2fit = polyval(p, logt);
ssresid = sum((logr2 - logr2fit).^2);
rsq = 1. - ssresid / ((Nstep - 1) * var(logr2));
r = sign(p(1)) * sqrt(rsq)
sa = sqrt(ssresid / ((Nstep - 2) * (Nstep - 1) * var(logt)))

plot(logt, polyval(p, logt), 'b');
%
%END





