function [x] = Lab9_generate_rw(Nstep)

%
% grow an RW from origin to Nstep steps. locations kept just for future use
%

x = zeros(2, Nstep + 1);

dr = [1 -1 0  0
      0  0 1 -1 ];
x(:,1) = [ 0
           0 ];

for j = 2:Nstep+1
   r = rand;
   if r <= 0.25
      x(:,j) = x(:,j-1)+dr(:,1);
   elseif r <= 0.5
      x(:,j) = x(:,j-1)+dr(:,2);
   elseif r <= 0.75
      x(:,j) = x(:,j-1)+dr(:,3);
   else
      x(:,j) = x(:,j-1)+dr(:,4);
   end
end
