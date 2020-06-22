# import numpy as np
# import matplotlib.pyplot as plt
#
#
# hbar = 1
# m = 1
# k = 1
# omega = np.sqrt(k / m)
# L = 1
# V0 = 10
# Vinf = 100000
# dpsi_0 = 1
# zero = 1e-11
# tol = 1e-8
#
#
# def e_ipw(n):
#     return (np.pi * hbar * n / L) ** 2 / (2 * m)
#
#
# def e_qho(n):
#     return hbar * omega * (n + 0.5)
#
#
# def V_wall(x):
#     if abs(x) > L:
#         return Vinf
#     elif abs(x) < L / 50:
#         return V0
#     else:
#         return 0
#
#
# def V_double_wall(x):
#     if -L/10 < x < -2*L/10:
#         return V0
#     if 3*L/10 < x < 4*L/10:
#         return V0
#     else:
#         return 0
#
#
# def V_ipw(x):
#     return 0 if abs(x) < L else Vinf
#
#
# def V_fpw(x):
#     return 0 if abs(x) < L else V0
#
#
# def V_step(x):
#     return V0 if x > L/2 else 0
#
#
# def V_qho(x):
#     return 0.5 * m * omega ** 2 * x ** 2
#
#
# def norm(psi, domain):
#     # trapezoidal rule
#     sum = 0
#     for i in range(np.size(psi) - 1):
#         f_i = (np.real(psi[i]) + np.imag(psi[i])) * (np.real(psi[i]) - np.imag(psi[i]))
#         f_ip1 = (np.real(psi[i + 1]) + np.imag(psi[i + 1])) * (np.real(psi[i + 1]) - np.imag(psi[i + 1]))
#         sum = sum + 0.5 * (f_i + f_ip1) * (domain[i+1]-domain[i])
#     return(psi / np.sqrt(sum))
#
#
# def solve(_y, _x,V, E):
#     n = np.size(_x)
#     for i in range(1, n - 1):
#         h = _x[i + 1] - _x[i]
#         _y[i + 1] = 2 * (m * (h/hbar)**2 * (V(_x[i]) - E) + 1) * _y[i] - _y[i - 1]
#     return(norm(_y,_x))
#
#
# def normsqr(psi):
#     return (np.real(psi) + np.imag(psi)) * (np.real(psi) - np.imag(psi))
#
#
# def obj_func(_y,_x,V,bval, E):
#     Psi = solve(_y,_x,V,E)
#     return Psi[-1] - bval
#
#
# def shoot(wavefunc,domain,V,bval, a, b):
#     # regula falsi
#     c = 0
#     while abs(b - a) > tol:
#         fa = obj_func(wavefunc, domain, V, bval, a)
#         fb = obj_func(wavefunc, domain, V, bval, b)
#         c = (a * fb - b * fa) / (fb - fa)
#         fc = obj_func(wavefunc, domain, V, bval, c)
#
#         if abs(fc) < zero:
#             return c
#         elif fa * fc > 0:
#             a = c
#         else:
#             b = c
#     return c
#
#
# def main():
#     L = 10
#     V = V_qho
#     domain = np.linspace(-L / 2, L / 2, 1000)
#     psi = 0*domain
#     psi[0],psi[1] = 0, dpsi_0*(domain[1]-domain[0])
#
#     pairs_qho = [(0,1),(1,2),(2,3),(3,4),(4,5),
#                  (5,6),(6,7),(7,8),(8,9),(9,10)]
#     # E = shoot(psi,domain,V_qho,0,2,3)
#     # print(E)
#     # # Es = np.linspace(0,40,100)
#     # # plt.plot(Es,[obj_func(psi,domain,V,0,e) for e in Es])
#     # # plt.show()
#     for pair in pairs_qho:
#         E = shoot(psi, domain, V_qho, 0, *pair)
#         Psi = solve(psi,domain,V_qho,E)
#         plt.plot(domain, E + norm(Psi, domain))
#     plt.fill_between(domain, [V(x) for x in domain], label="V(x)", color='black', alpha=0.5)
#     plt.grid()
#     plt.legend()
#     plt.show()
# main()

import numpy as np

hbar = 1.0
m = 1.0
k = 1.0
omega = np.sqrt(k / m)
inf = 1000.0
zero = 1.0e-12
eps = 1.0e-6
Vinf = 100000.0
dpsi_0 = 1.0


def main():
    helpstring = '''qho:         Quantum Harmonic Oscillator\n
                    infsqrwell:  Infinite Square Well\n
                    finsqrwell:  Finite Square Well\n'''


    x_i, x_f, num_x = input("Input domain (start stop N)\n").split(" ")
    x_i, x_f, num_x = float(x_i), float(x_f), int(num_x)
    dx = (x_f-x_i)/(num_x-1)
    x = np.zeros(num_x)
    x[0] = x_i
    for j in range(1,num_x):
        x[j] = x[j-1] + dx
    print(f"set x = <{x[0]}, ..., {x[-1]}>")
    psi = np.zeros(num_x)

    lbc,rbc = input("Input boundary conditions (usually '0 0')\n").split(" ")
    lbc, rbc = float(lbc), float(rbc)
    print(f"set lbc, rbc = {lbc}, {rbc}")

    E_i, E_f, num_E = input("Input energy range (start stop N)\n").split(" ")
    E_i, E_f, num_E = float(E_i), float(E_f), int(num_E)
    dE = (E_f-E_i)/(num_E-1)
    E = np.zeros(num_E)
    E[0] = E_i
    for j in range(1,num_E):
        E[j] = E[j-1] + dE
    print(f"set E = <{E[0]}, ..., {E[-1]}>")

    method = input("Input a potential, type 'help' to get a list of available options\n")
    if method == "help":
        print(helpstring)
        return 0

    with open("../../data/Chapter10/output.dat", 'w') as file:
        file.write(r"# E \psi_f-bc\n")

        if method == "qho":
            print("You chose the quantum harmonic oscillator")

            for i in range(num_E):
                sum = 0
                psi[0] = lbc
                psi[1] = dpsi_0*dx

                for j in range(1,num_x-1):
                    psi[j+1] = 2*(m*(dx/hbar)**2*(0.5*m*omega**2*x[j]**2-E[i])+1)*psi[j] - psi[j-1]

                # Normalize psi
                for n in range(num_x-1):
                    f_i = (np.real(psi[n]) + np.imag(psi[n])) * (np.real(psi[n]) - np.imag(psi[n]))
                    f_ip1 = (np.real(psi[n + 1]) + np.imag(psi[n + 1])) * (np.real(psi[n + 1]) - np.imag(psi[n + 1]))
                    sum = sum + 0.5 * (f_i + f_ip1) * dx
                for n in range(num_x):
                    psi[n] = psi[n]/np.sqrt(sum)

                file.write(f"{E[i]} {psi[-1] - rbc}\n")
            print("Done!")

        # if(method == "infsqrwell"){
        #     long double L;
        #     long double sum;
        #     cout << "You chose the infinite square well" << endl;
        #     cout << "As a reminder, the ISW is centered about zero such that the domain is (-L/2, L/2)" << endl;
        #
        #     cout << "L:";
        #     cin >> L;
        #     cout << "set L = " << L << endl;
        #
        #     for(unsigned int i = 0; i<num_E;i++){
        #         sum = 0.0;
        #         psi[0] = lbc;
        #         psi[1] = dpsi_0*dx;
        #
        #         for(unsigned int j = 1; j<num_x-1;j++){
        #             psi[j+1] = 2*((pow(dx*hbar,2)/(2.0*m))*((fabs(x.at(j)) < L/2.0 ? 0.0 : Vinf)-E[i])+1)*psi[j] - psi[j-1];
        #         }
        #
        #         //Normalize psi
        #         for(unsigned int k=0;k<num_x-1;k++) {
        #             f_i = (real(psi[k]) + imag(psi[k])) * (real(psi[k]) - imag(psi[k]));
        #             f_ip1 = (real(psi[k + 1]) + imag(psi[k + 1])) * (real(psi[k + 1]) - imag(psi[k + 1]));
        #             sum = sum + 0.5 * (f_i + f_ip1) * dx;
        #         }
        #         for(unsigned int l=0;l<num_x;l++) {
        #             psi[l] = psi[l]/sqrt(sum);
        #         }
        #
        #         file << E[i] << " " << psi.back() - rbc << endl;
        #     }
        #     cout << "Done!" <<endl;
        # }
        #
        # if(method == "finsqrwell"){
        #     long double L;
        #     long double V0;
        #     long double sum;
        #     cout << "You chose the finite square well" << endl;
        #     cout << "As a reminder, the FSW is centered about zero such that the domain is (-L/2, L/2)" << endl;
        #
        #     cout << "L:";
        #     cin >> L;
        #     cout << "set L = " << L << endl;
        #
        #     cout << "V_0:";
        #     cin >> V0;
        #     cout << "set V_0 = " << V0 << endl;
        #
        #     for(unsigned int i = 0; i<num_E;i++){
        #         sum = 0.0;
        #         psi[0] = lbc;
        #         psi[1] = dpsi_0*dx;
        #
        #         for(unsigned int j = 1; j<num_x-1;j++){
        #             psi[j+1] = 2*((pow(dx*hbar,2)/(2.0*m))*((fabs(x.at(j)) < L/2.0 ? 0.0 : V0)-E[i])+1)*psi[j] - psi[j-1];
        #         }
        #
        #         //Normalize psi
        #         for(unsigned int k=0;k<num_x-1;k++) {
        #             f_i = (real(psi[k]) + imag(psi[k])) * (real(psi[k]) - imag(psi[k]));
        #             f_ip1 = (real(psi[k + 1]) + imag(psi[k + 1])) * (real(psi[k + 1]) - imag(psi[k + 1]));
        #             sum = sum + 0.5 * (f_i + f_ip1) * dx;
        #         }
        #         for(unsigned int l=0;l<num_x;l++) {
        #             psi[l] = psi[l]/sqrt(sum);
        #         }
        #
        #         file << E[i] << " " << psi.back() - rbc << endl;
        #     }
        #     cout << "Done!" <<endl;
        # }
        return 0

main()