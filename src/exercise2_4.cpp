/*---------------------------------------------------------------------------*\
  ========          |  Purdue Physics 580 - Computational Physics
  \\                |  Chapter 2 - Exercise 3
   \\               |  
   //               |  Author: Ethan Knox
  //                |  Website: https://www.github.com/ethank5149
  ========          |  MIT License
-------------------------------------------------------------------------------
License
Copyright 2020 Ethan Knox

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
\*---------------------------------------------------------------------------*/

// Including
/*****************************************************************************/
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "../include/xtensor/xarray.hpp"
#include "../include/xtensor/xio.hpp"
#include "../include/xtensor/xview.hpp"
#include "../include/xtensor/xmath.hpp"
#include "../include/xtensor/xnorm.hpp"
#include "../include/xtensor/xfunction.hpp"
#include "../include/xtensor/xoperation.hpp"
/*****************************************************************************/

// Using
/*****************************************************************************/
using namespace std;
using namespace xt;
/*****************************************************************************/

// Global Definitions
#define g 9.81     //Gravitational Acceleration [m/s^2]
#define num_eqns 2 // Number of equations

// Struct Declarations
struct parameters
{
    double m;       // Mass
    double C;       // Drag Coefficient
    double rho;     // Air density
    double A;       // Surface area
    double P;       // Power
    double v0;      // Initial Velocity
    double theta;   // Angle of incline
    double dt;      // Time step
    double tf;      // Final time
    unsigned int N; // Number of time-steps
} params;

// Function Declarations
/*****************************************************************************/
/*(params, t, X)*/
void initialize(parameters &, xarray<double> &, xarray<double> &);
void calculate(parameters &, xarray<double> &, xarray<double> &);
void store(parameters &, xarray<double> &, xarray<double> &);

int main()
{
    xarray<double> t = zeros<double>({100});    // Time (x)array
    xarray<double> X = zeros<double>({100, 2}); // Data (x)array

    initialize(params, t, X);
    calculate(params, t, X);
    store(params, t, X);
}

void initialize(parameters &params, xarray<double> &t, xarray<double> &X)
{
    double angle;
    double v0;

    cout << "Running 'exercise2_4.exe'...\n\n";
    cout << "Input Parameters\n";
    cout << "================\n";
    cout << "m = ";
    cin >> params.m;
    cout << "C = ";
    cin >> params.C;
    cout << "rho = ";
    cin >> params.rho;
    cout << "theta (degrees) = ";
    cin >> angle;
    params.theta = M_PI * angle / 180.0; // Convert to radians
    cout << "A = ";
    cin >> params.A;
    cout << "P = ";
    cin >> params.P;
    cout << "dt = ";
    cin >> params.dt;
    cout << "t_f = ";
    cin >> params.tf;

    params.N = (unsigned int)ceil(params.tf / params.dt);
    // Resize our data (x)arrays to the correct size
    X.resize({params.N + 1, 2});
    X.fill(0.);
    t.resize({params.N + 1});
    t.fill(0.);

    cout << "v0 = ";
    cin >> v0;
    X(0, 0) = v0 * cos(params.theta);
    X(0, 1) = v0 * sin(params.theta);
    cout << "Done.\n\n";
    return;
}

void calculate(parameters &params, xarray<double> &t, xarray<double> &X)
{
    xarray<double> F_Net;                                     // Applied force on the bike
    xarray<double> F_g{-params.m * g * sin(params.theta), 0}; // Gravitational Force
    xarray<double> F_d;                                       // Drag force
    double b = -0.5 * params.C * params.rho * params.A;       // Calculate the aggregate drag coefficient only once in advance
    cout << "Running Simulation...\n";

    for (unsigned int i = 0; i <= params.N - 1; i++)
    {
        // Forward Euler Method
        F_d = b * norm_l2(row(X, i))() * row(X, i);
        F_Net = params.P * row(X, i) / norm_sq(row(X, i))() + F_g + F_d;
        row(X, i + 1) = row(X, i) + (F_Net / params.m) * params.dt;
        t(i + 1) = t(i) + params.dt;
    }
    cout << "Done.\n\n";
    return;
}

void store(parameters &params, xarray<double> &t, xarray<double> &X)
{
    // Declare and open the output fine, then write data headers
    cout << "Writing to file...\n\n";
    ofstream f("../data/exercise2_4_output.csv");
    f << "m, C, rho, theta, A, P, dt, t_final, vx0, vy0\n";
    f << params.m << ", " << params.C << ", " << params.rho << ", " << params.theta << ", " << params.A << ", " << params.P << ", " << params.dt << ", " << params.tf << ", " << X(0, 0) << ", " << X(0, 1) << "\n";
    f << "t, vx, vy\n";

    // Write data to output file
    for (unsigned int i = 0; i <= params.N; i++)
    {
        f << t(i) << ", " << X(i, 0) << ", " << X(i, 1) << "\n";
    }
    cout << "Done.\n\n";
    return;
}
// 70 0.5 1.225 0.33 400 0.1 200