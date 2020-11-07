% 
% Ising model in two dimensions on a square lattice
% Shows spin flips in color, no statistics or graphs
%
% code by Hisao Nakanishi, modified by Denes Molnar
%
clear;
%
% Initialize parameters (NxN lattice)
%
N   = input('Lattice size N: ');
T   = input('Temperature [J/k_B units]: ');
H   = input('Magnetic field [J/mu units]: ');
MCS = input('Monte Carlo steps per spin: ');
c   = input('Initial spin config (1: all up, -1: all down, 2: random): ');
out = input('output file for magnetization/energy time series: ', 's');
fprintf('N = %d, T = %0.5f (J/k_B), H = %0.5f (J/mu)\n', N ,T, H);
%
% Set initial conditions
%
%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);
%
% set spin directions
if c == 1
   spin = ones(N, N);
elseif c == -1
   spin = ones(N, N).*(-1);
else
   for i = 1:N
      for j = 1:N
         if rand <= 0.5
            spin(i,j) = 1;
         else
            spin(i,j) = -1;
         end
      end
   end %i
end %c
% compute initial energy and magnetization
m = 0;
E = 0;
for i = 1:N
   for j = 1:N
      m = m + spin(i,j);
      % find the net alignment of the nearest neighbors of spin i,j
      % note periodic boundary conditions
      sum = spin(mod(i,N)+1,j)     + spin(i,mod(j,N)+1) ...
            + spin(N-mod(1-i,N),j) + spin(i,N-mod(1-j,N));
      E = E - spin(i,j) * sum * 0.5 - H * spin(i,j);
   end
end

%
% do Monte Carlo updates
% 
% before first sweep, visualize initconds and write output (if requested)
results(1,:) = [1 m/N^2 E/N^2];
step = 1;
fprintf('MCS=%d: M = %0.5f, E = %0.5f \n', step - 1, results(step,2), results(step,3));
if out
   dlmwrite(out, results(step,:), '-append', 'newline', 'pc', 'delimiter', '\t');
end
% display initial spin configurations
clf;
jb = 0;
jr = 0;
for i = 1:N
   for j = 1:N
      if spin(i,j) == -1
         jb = jb + 1;
         blue(jb,:) = [i j];
      else
         jr = jr + 1;
         red(jr,:) = [i j];
      end
   end
end
symbolSize = 800 / (N/20)^2;   % adjust to your screen
if jb > 0
   scatter(blue(:,1), blue(:,2), symbolSize, 'b', 'Filled');
end
hold on;
if jr > 0
   scatter(red(:,1), red(:,2), symbolSize, 'r');
end
hold on;
display('Hit Enter to continue')
pause;
% do rest of the MC steps
nskip = 1; % frequency of diagnostic output to stdout
%nskip = 10;
for step = 2:MCS+1
   [spin,E,m] = Lab11_calculate_spin(spin, N, T, H, E, m, symbolSize);
   results(step,:) = [step m/N^2 E/N^2];
   s = sprintf('MCS=%d: M = %0.5f, E = %0.5f \n', step-1, results(step,2), results(step,3));
   title(s);
   %pause(0.0001);
   %pause();
   if mod(step - 1, nskip) == 0
      fprintf('MCS=%d: M = %0.5f, E = %0.5f \n', step-1, results(step,2), results(step,3))
   end
   if out
      dlmwrite(out, results(step,:), '-append', 'newline', 'pc', 'delimiter', '\t');
   end
end
%
%END

