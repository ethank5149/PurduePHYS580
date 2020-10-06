# Gnuplot script file for plotting data in file "../data/poincare.dat"
# Plot with:
# terminal```
# >gnuplot
# >load 'poincare.gnuplot'
# ```

set autoscale
set xtic auto
set ytic auto
set title "Poincare Section"
set xlabel "\theta []"
set ylabel "\omega []"
#      set xr [-3.15:3.15]
#      set yr [-1.5:1.5]
plot "../data/poincare.dat" using 2:3 with points