//
// Created by ethan on 6/16/2020.
//

#ifndef PURDUE_PHYS_580_PROBLEM3_18_H
#define PURDUE_PHYS_580_PROBLEM3_18_H
#endif //PURDUE_PHYS_580_PROBLEM3_18_H


#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <iomanip>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xoperation.hpp"

// Using
/*****************************************************************************/
using namespace std;
using namespace xt;
/*****************************************************************************/

struct Problem3_18
{
    const long double l = 9.8;
    const long double g = 9.8;
    const long double q = 0.5;
    const long double Omega_D = 2.0/3.0;

    const int num_F_D = 1000;

    const int num_points = 400;
    const int transient_wait = 300;
    const int point_density = 1000;

    const long double theta0 = 0.2;
    const long double dtheta0 = 0.0;

    const long double F_D_i = 1.35;
    const long double F_D_f = 1.5;
    const long double dF_D = (F_D_f - F_D_i)/num_F_D;

    const long double t_f = 2*M_PI*num_points/Omega_D;
    const long double N = point_density*num_points;
    const long double dt = t_f / N;

    void run()
    {
        xarray<long double> x = {theta0, dtheta0};
        long double t = 0.0;
        long double F_D = F_D_i;

        calculate(t, x, F_D);
    }


    static void constrain_domain(long double &x)
    {
        if(x < -M_PI) {
            x = x + 2 * M_PI;
        }
        else if(x > M_PI)
        {
            x = x - 2 * M_PI;
        }
    }


    [[nodiscard]] xarray<long double> f(const long double &t, const xarray<long double> &x, const long double &F_D) const
    {
        xarray<long double> df = {x(1), -(g/l)*sin(x(0))-q*x(1)+F_D*sin(Omega_D*t)};
        return df;
    }


    void calculate(long double &t, xarray<long double> &x, long double &F_D) const
    {
        xarray<long double> k1;
        xarray<long double> k2;
        xarray<long double> k3;
        xarray<long double> k4;

        cout << "Running Simulation...\n";
        ofstream file("../data/Problem3_18.dat");
        file << "# F_D X\n";
        file << setprecision(numeric_limits<long double>::digits10 + 1);

        for (unsigned int j = 0; j<num_F_D;j++){
            int num_cycles = 0;
            for (unsigned int i = 0; i < num_points*point_density - 0*1; i++)
            {
                if (i%point_density==0){
                    num_cycles++;
                    if (num_cycles>transient_wait){
                        file << F_D << " " << x(0) << "\n";
                    }
                }

                // RK4
                k1 = dt * f(t, x, F_D);
                k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, F_D);
                k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2, F_D);
                k4 = dt * f(t + dt, x + k3, F_D);
                x = x + (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
                constrain_domain(x(0));
                t += dt;
            }
            F_D += dF_D;
        }
        cout << "Done.\n\n";
    }
} p3_18;
