from lib.DSolve import euler, rk2a, rk2b, rk4, heun

from pylab import *

def f(t, x):
    return np.array([x[1], -np.sin(x[0])])

a, b = (0.0, 10.0)
x0 = [np.pi/2.01, 0.0]

n = 100
t = np.linspace(a, b, n)

from scipy.integrate import solve_ivp

# compute various numerical solutions
x_euler = euler(f, x0, t)
x_heun = heun(f, x0, t)
x_rk2a = rk2a(f, x0, t)
x_rk2b = rk2b(f, x0, t)
x_rk4 = rk4(f, x0, t)
x = solve_ivp(f, (a, b), x0, t_eval=t).y

#   figure( 1 )
subplot(1, 2, 1)
plot(t, x_euler[0], label="Euler")
plot(t, x_heun[0], label="Heun")
plot(t, x_rk2a[0], label="RK2a")
plot(t, x_rk2b[0], label="RK2b")
plot(t, x_rk4[0], label="RK4")
plot(t, x[0], 'k--', label="solve_ivp")
xlabel(r'$t$')
ylabel(r'$x$')
legend()

#   figure( 2 )
subplot(1, 2, 2)
plot(t, x_euler[0] - x[0], label="Euler")
plot(t, x_heun[0] - x[0], label="Heun")
plot(t, x_rk2a[0] - x[0], label="RK2a")
plot(t, x_rk2b[0] - x[0], label="RK2b")
plot(t, x_rk4[0] - x[0], label="RK4")
xlabel('$t$')
ylabel('$x - x^*$')
legend()
show()
