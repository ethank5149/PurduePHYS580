%
% Self-avoiding walk in 2 dimensional square lattice
%

clear all;
clf;

% input physical parameters
%
Nsteps  = input('maximum number of steps: ');
n1     = input('number of walks at step 1: ');
cutoff = input('min fraction of surviving walks below which to cut off: ');


% initialize RNG, default one is good enough
%
%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);


% initialize variables and arrays
%
%plot_flag = 1;
nwalks = zeros(1, Nsteps);
x2ave = zeros(1, Nsteps);
x     = zeros(2, Nsteps + 1);

% maintain a lattice to quickly find whether the site has already been visited
%
latt = zeros(2*Nsteps+1, 2*Nsteps+1);

% attempt to generate SAW's 
%
maxstep = 1;
for i = 1:n1
   % generate a SAW
   [x, mstep] = Lab9_generate_saw(latt, Nsteps);
   if mstep > maxstep
      maxstep = mstep;  % keep the max length of generated SAW's
   end
   % plot path (or comment out next 3 lines for speed)
   %plot(x(1,1:mstep+1), x(2,1:mstep+1), 'b', 'LineWidth', 3);
   %axis([-Nsteps Nsteps -Nsteps Nsteps]);
   %pause;
   % statistics for one SAW
   for j = 1:mstep
      nwalks(j) = nwalks(j) + 1;  % count survivors (SAWs have high attrition)
      x2ave(j) = x2ave(j) + x(1,j+1)^2 + x(2,j+1)^2;
   end
end


% get <r^2> and standard deviation
%
jmax = 0;
for j=1:maxstep
   if nwalks(j) > cutoff * nwalks(1);  % only keep if sufficient stats
      x2ave(j) = x2ave(j) / nwalks(j);
      jmax = j;
   else
      break;  % otherwise stop at this length (meaningless accuracy beyond it)
   end
end


% plot <r^2> and its standard deviation vs t data points in log-log plot
%
t = (1:jmax);
logt = log10(t);
logr2 = log10(x2ave(1:jmax));

clf;
scatter(logt, logr2, 15, 'r', 'x');

title('SAW in d=2')
xlabel('log_{10} t');
ylabel('log_{10} <r^2> (red), log10 SD');

hold on;


% linear least squares fit
%
last = input('up to how many steps to fit? ')

logt = log10(t(1:last));
logr2 = log10(x2ave(1:last));
p = polyfit(logt,logr2,1)

logr2fit = polyval(p, logt);
ssresid = sum((logr2 - logr2fit).^2);
rsq = 1. - ssresid / ((jmax-1) * var(logr2));
r = sign(p(1)) * sqrt(rsq)
sa = sqrt(ssresid / ((jmax-2) * (jmax-1) * var(logt)))

plot(logt, polyval(p, logt), 'b');
%
%END





