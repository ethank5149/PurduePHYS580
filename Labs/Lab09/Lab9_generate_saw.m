function [x, mstep] = Lab9_generate_saw(latt, Nsteps)

% grow a SAW from origin to Nsteps steps, if possible
%

x = zeros(2, Nsteps + 1);

dr = [1 -1 0  0
      0  0 1 -1 ];
x(:,1) = [ 0
           0 ];

% track occupancy on a grid
latt(Nsteps + 1, Nsteps + 1) = 1;    % origin is occupied

% first step always in x direction (without loss of generality)
x(:,2) = [ 1
           0 ];
latt(Nsteps + 2, Nsteps + 1) = 1;    % (1,0) is occupied
nodir = 2;

mstep = 1;
for j = 3:Nsteps+1
   % generate next direction, ignore stepping back
   r = floor(rand * 3 + 1); %
   if r >= nodir
       r = r + 1;
   end
   % store new position
   x(:,j) = x(:,j-1) + dr(:,r);
   % if (x,y) is occupied, terminate walk
   if latt(x(1,j) + Nsteps + 1, x(2, j) + Nsteps + 1) > 0
      break;
   % otherwise, update length, backwards direction, and lattice
   else
      mstep = mstep + 1;
      nodir = mod(r, 2) + 2 * floor((r-1) / 2) + 1;
      latt(x(1,j) + Nsteps + 1, x(2,j) + Nsteps + 1) = 1;
   end
end

for j = 3:mstep+1
    latt(x(1,j) + Nsteps + 1, x(2,j) + Nsteps + 1) = 0;  % restore empty lattice
end

