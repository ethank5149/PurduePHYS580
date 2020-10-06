# Gnuplot script file for plotting data in file "../data/bifurcation.dat"
# Plot with:
# terminal```
# >gnuplot
# >load 'bifurcation.gnuplot'
# ```

set autoscale
set xtic auto
set ytic auto
set title "Bifurcation Diagram"
set xlabel "Amplitude []"
set ylabel "\theta []"
#      set xr [-3.15:3.15]
#      set yr [-1.5:1.5]
plot "../data/bifurcation.dat" using 1:2 with points