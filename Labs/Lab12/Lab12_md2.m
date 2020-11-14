% molecular dynamics in two dimensions
%
% Matlab version with keyboard interactions added by H. Nakanishi
%    (based on original TB version by N. Giordano)
% modified by Denes Molnar Nov 2019
%
clear;
clf;
%
% Read parameters
%
Npart  = input('# of particles: ');
L      = input('side length L [units of sigma]: ');
dt     = input('time step dt [units given on p.274]: ');
n_plot = input('plot/record after this many steps: ');
vmax   = input('max initial speed: ');
dmax   = input('max initial relative displacement: ');
out    = input('file name for thermodynamics output: ');
disp('Type c or d (clear/remove green tracks), p to pause, or q to plot time series and end.\n'); 
disp('type + to increase speed 50% or - to decrease 50%, or \n');
disp('type N [N=1,...,4] to increase speed by 10N%. \n');
disp('Hit any key to start...');

x  = zeros(2,Npart);
y  = zeros(2,Npart);
vx = zeros(Npart,1);
vy = zeros(Npart,1);
xp = zeros(Npart,1);
yp = zeros(Npart,1);

%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);

%
% Generate initial conditions
%
sqN = sqrt(Npart);
if (sqN == floor(sqN))
   grid = L / sqN;
else
   grid = L / floor(sqN + 1);
end

n = 0;
i = 0;
while i < L       % arrange particles on a roughly square lattice
   j = 0;          % to keep them apart initially
   while j < L
      n = n + 1;
      if n <= Npart
         x(2,n) = i + 0.5 * grid + dmax * (rand-0.5) * grid * sqrt(2);
         y(2,n) = j + 0.5 * grid + dmax * (rand-0.5) * grid * sqrt(2);
         % random initial velocities
         vx(n) = vmax * (rand-0.5) * sqrt(2);
         vy(n) = vmax * (rand-0.5) * sqrt(2);
         x(1,n) = x(2,n) - vx(n) * dt;
         y(1,n) = y(2,n) - vy(n) * dt;
      end
      j = j + grid;
   end
   i = i + grid;
end %i

t = 0;

% initial display of particles
%
symSize = 200; %symbol size (area)
h3 = scatter(x(2,3:Npart), y(2,3:Npart), symSize, 'r', 'Filled');
hold on;
h1 = scatter(x(2,1:1), y(2,1:1), symSize, 'b', 'Filled');
h2 = scatter(x(2,2:2), y(2,2:2), symSize, 'g', 'Filled');

xlabel('x');
ylabel('y');
xlim([0 L]);
ylim([0 L]);
hold on;
pause;

drg = 1;    % whether to draw green circles for positions
for i = 1:Npart
   xp(i) = x(2,i);
   yp(i) = y(2,i);
end
set(gcf, 'CurrentCharacter', '@'); % set to a dummy character



% 
% Evolve with Newtonian dynamics
%
i = 0;
j = 0;
n_p = 0;
r2 = 0;
x1 = 0;
y1 = 0;
x2 = x(2,2) - x(2,1);
y2 = y(2,2) - y(2,1);

while 1    % infinite loop

   [x,y,vx,vy,r2,rp2,x1,y1,x2,y2] = Lab12_updatemd(x, y, vx, vy, Npart, L, dt, x1, y1, x2, y2);

   t = t + dt;
   j = j + 1; 		% use to keep track of how often to plot on screen
   if j >= n_plot		% and record values for later plotting
      if drg == 1
         scatter(xp(1:Npart), yp(1:Npart), symSize, 'g');
      end
      %set(h1, 'XData', x(2,1), 'YData', y(2,1));
      %set(h2, 'XData', x(2,2), 'YData', y(2,2));
      %set(h3, 'XData', x(2,3:Npart), 'YData', y(2,3:Npart));
      %drawnow;
      for k = 1:Npart
         xp(k) = x(2,k);
         yp(k) = y(2,k);
      end

      j = 0;
      i = i + 1;

      % calculate and record time, energy, and temperature
      [E,Epot,T] = Lab12_calc_energy(x, y, vx, vy, Npart, L);
      s = sprintf('t = %0.5f, energy = %0.5e, temperature = %0.5e \n', t, E, T);
      title(s);

      thermo(i,:) = [t, E, T, r2, rp2];
      if out
         dlmwrite(out, thermo(i,:), '-append', 'newline', 'pc', 'delimiter', '\t');
      end
      n_p = n_p + 1;
      velRescale = 1;
      %pause(0.001);
      figure(gcf);
      key = get(gcf, 'CurrentCharacter');
      %key = input('input: ', 's');
      switch key
         case 'q'
            break;
         case {'c','d'}
            clf;
            h3 = scatter(x(2,3:Npart), y(2,3:Npart), symSize, 'r', 'Filled');
            hold on;
            h1 = scatter(x(2,1), y(2,1), symSize, 'b', 'Filled');
            h2 = scatter(x(2,2), y(2,2), symSize, 'g', 'Filled');
            xlabel('x');
            ylabel('y');
            xlim([0 L]);
            ylim([0 L]);
            hold on;
            drawnow;
            if key == 'd'
                drg = 0;
            end
         case 'u'
             drg = 1;
         case 'p'
            pause;
         case '+'   % heat up v -> v * 1.5
            velRescale = 1.5;
         case '-'   % cool v -> v * 0.5
            velRescale = 0.5;
         case '1'  % heat up v -> v * 1.1
            velRescale = 1.1;
         case '2'  % heat up v -> v * 1.2
            velRescale = 1.2;
         case '3'  % heat up v -> v * 1.3
            velRescale = 1.3;
         case '4'  % heat up v -> v * 1.4
            velRescale = 1.4;
      end %switch
      set(gcf, 'CurrentCharacter', '@');
      % rescale velocities, if requested
      if velRescale ~= 1
         for k = 1:Npart
            x(1,k) = x(2,k) - velRescale * (x(2,k) - x(1,k));
            y(1,k) = y(2,k) - velRescale * (y(2,k) - y(1,k));
         end
      end
   end %
   set(gcf, 'CurrentCharacter', '@');
end %while

% now finished      prepare for plotting final results
disp('hit any key to display time series ...')
pause;

lw = 3;

clf;
plot(thermo(:,1), thermo(:,2), 'r', 'LineWidth', lw)
title('Energy time series');
xlabel('Time (in L-J units)');
ylabel('Total energy (in epsilon)');
disp('hit any key to continue')
pause;

clf;
plot(thermo(:,1), thermo(:,3), 'b', 'LineWidth', lw)
title('Temperature time series');
xlabel('Time (in L-J units)');
ylabel('Temperature (in epsilon/k_B)');
disp('hit any key to continue')
pause;

clf;
plot(thermo(:,1), thermo(:,4), 'g', 'LineWidth', lw)
title('Tagged particle square displacement time series');
xlabel('Time (in L-J units)');
ylabel('r^2 (in sigma^2)');
disp('hit any key to continue')
pause;

clf;
plot(thermo(:,1), thermo(:,5), 'g', 'LineWidth', lw)
title('Tagged pair square distance time series');
xlabel('Time (in L-J units)');
ylabel('dr^2 (in sigma^2)');
%
%
%END
