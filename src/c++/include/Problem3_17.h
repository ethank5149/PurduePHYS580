//
// Created by ethan on 6/14/2020.
//

#ifndef PURDUE_PHYS_580_PROBLEM3_17_H
#define PURDUE_PHYS_580_PROBLEM3_17_H

#endif //PURDUE_PHYS_580_PROBLEM3_17_H

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

// Global Definitions
const long double l = 9.8;
const long double g = 9.8;
const long double q = 0.5;
const long double Omega_D = 2.0/3.0;
const long double F_D = 1.2;

const int num_points = 1000000;
const int point_density = 1000;
const long double theta0 = 0.2;
const long double dtheta0 = 0.0;

const long double t_f = 2*M_PI*num_points/Omega_D;
const long double N = point_density*num_points;
const long double dt = t_f / N;


// Function Declarations
xarray<long double> f(const long double &, const xarray<long double> &);
void constrain_domain(long double &);
void calculate(long double &, xarray<long double> &);


void run_problem3_17()
{
    xarray<long double> x = {theta0, dtheta0};
    long double t = 0.0;

    calculate(t, x);
}


void constrain_domain(long double &x)
{
    if(x < -M_PI) {
        x = x + 2 * M_PI;
    }
    else if(x > M_PI)
    {
        x = x - 2 * M_PI;
    }
}


xarray<long double> f(const long double &t, const xarray<long double> &x)
{
    xarray<long double> df = {x(1), -(g/l)*sin(x(0))-q*x(1)+F_D*sin(Omega_D*t)};
    return df;
}


void calculate(long double &t, xarray<long double> &x)
{
    xarray<long double> k1;
    xarray<long double> k2;
    xarray<long double> k3;
    xarray<long double> k4;

    cout << "Running Simulation...\n";
    ofstream file("../data/Problem3_17.csv");
    file << "X, dX\n";
    file << setprecision(numeric_limits<long double>::digits10 + 1);
    file << x(0) << ", " << x(1) << "\n";

    for (unsigned int i = 0; i < num_points*point_density - 1; i++)
    {
        // RK4

        k1 = dt * f(t, x);
        k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1);
        k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2);
        k4 = dt * f(t + dt, x + k3);
        x = x + (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
        constrain_domain(x(0));
        t += dt;

        if((i+1)%point_density == 0)
        {
            file << x(0) << ", " << x(1) << "\n";
        }
    }
    cout << "Done.\n\n";
}
