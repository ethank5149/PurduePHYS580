
% ***********************************************************************
%     Program to generate percolation clusters (label tree version)
% ***********************************************************************

% initialize limits
maxedge = 10000;

%initialize parameters
p       = input('Site occupation probability (p): ');
Nedge   = input('lattice edge length (L): ');
Ntrials = input('number of realizations: ');
clustFile = input('cluster output file name: ', 's');
distFile  = input('size distribution output file name: ', 's');
Nedge   = min(Nedge, maxedge);

% start the calculations!
%s = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(s);
for i = 1:Ntrials
   t0 = tic; % keep track of execution time
   % generate one percolation realization
   [list,list2,numb,nmax,nc,nna,nnb,vc] = Lab10_make2a(p, Nedge);
   cpu = toc(t0)
   fprintf('finished trial# = %d, p = %0.5f, edge = %d, cpu = %0.5f \n', i, p, Nedge, cpu);
   fprintf('total #sites = %d, max size = %d, #clusters = %d \n', numb, nmax, nc);
   %
   % save results in files if filename(s) were provided
   %
   if clustFile      % site list output overwrites file (last realization remains)
      dlmwrite(clustFile, transpose(list(:,1:nmax)), 'newline', 'pc', 'delimiter', '\t');
   end
   if distFile       % cluster statistics appends to file
      nnc = transpose( [nna(1:vc)
                        nnb(1:vc)] );
      dlmwrite(distFile, nnc, '-append', 'newline', 'pc', 'delimiter', '\t', 'precision', '%d');
   end
   % display the maximal cluster generated in this realization
   % comment out this block if collecting data for analysis and no pics needed
   clf;
   symbolSize = (200 / Nedge)^2 * 40;   % match the value "40" to screen resolution
   scatter(list(1,1:nmax), list(2,1:nmax), symbolSize, 'r', 'filled', 's');
   hold on;
   scatter(list2(1,1:numb-nmax), list2(2,1:numb-nmax), symbolSize, [0.5 0.5 0.5], '.');
   % calculate and show statistics: perc=percolation prob., susc=susceptibility
   susc = 0;
   perc = 0;
   for i = 1:vc-1
      perc = perc + nnb(i) * nna(i);
      susc = susc + nnb(i) * nna(i)^2;
   end
   perc = 1 - perc / numb;      % FIXME: taking nmax/numb would be quicker here...
   susc = susc / Nedge^2;
   fprintf('P(p) = %0.5f, S(p) = %0.5f \n', perc, susc);
   % pause for keystroke - comment this out if collecting data only
   input('Press enter to continue');
   %
end
%
%END  
