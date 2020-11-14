% Monte Carlo integration - evaluates f(x) at randomly sampled x
%

clear;
clf;

% integrand f(x), defined as an "anonymous function" in Matlab
%
f = @(x) sqrt(4. - x*x);

% read computational parameters
%
n = input('Number of random x values per trial = ');
Ntrials = input('Number of independent trials = ');

% initialize RNG 
% - Matlab uses the Mersenne Twister algorithm by default, which is good enough for us
%
%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);



% plot the integrand function
%
x1 = 0;
x2 = 2;
n1 = 1000;
dx = (x2 - x1) / n1;
x = zeros(n + 1, 1);
y = zeros(n + 1, 1);
for i = 1:(n1 + 1)
   x(i) = x1 + (i - 1) * dx;
   y(i) = f(x(i));
end
plot(x, y, 'b');
hold on;
xlabel('x');
ylabel('y');


% plot the MC integration progress in colors - turn plots off for speed
%
col(1) = 'r';
col(2) = 'm';
col(3) = 'g';
col(4) = 'b';
col(5) = 'y';
sum = 0;
sum2 = 0;
for j = 1:Ntrials
   r = 0;
   title(['Monte Carlo Integration I, trial = ',num2str(j)]);
   for i = 1:n
      t = rand * (x2 - x1) + x1;
      v = f(t);
      xx(i) = t;
      yy(i) = v;
      r = r + v;
   end
   %% COMMENT THIS BLOCK OUT FOR SPEED
   if (j == 1 || mod(j, 51) == 0);  % only plotting every 51st trial
      k = mod(j, 5) + 1;
      scatter(xx, yy, col(k), '.');
      pause(0.1);
   end
   %% UNTIL HERE
   s = r * (x2 - x1) / n;
   sum = sum + s;
   sum2 = sum2 + s^2;
end

% simple statistics in command window. "err" refers to the standard error of the mean
%
mean = sum / Ntrials;
var = sum2 / Ntrials - mean^2;
sd = sqrt(var);
err = sd / sqrt(Ntrials);
X = sprintf(' n = %d, trials = %d, mean = %0.5f, sd = %0.5f, err = %0.5f', ...
            n, Ntrials, mean, sd, err);
disp(X);

% compare to numerical integration by quadrature
% - Matlab has a routine that does just that
%
v = quad(f, 0, 2);
Y = sprintf(' Matlab Quadrature Result = %0.10f', v);
disp(Y);

