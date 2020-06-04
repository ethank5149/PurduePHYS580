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
//#define g 9.81            // Gravitational Acceleration [m/s^2]
//#define rho_air 1.225     // Density of air [kg/m^-3]
//#define rho_water 1000    // Density of water [kg/m^-3]
//#define eta_air 2.0e-5    // Viscosity of air
//#define eta_water 1.0e-3  // Viscosity of water
#define w_shoulders 0.411 // Average shoulder width

// Struct Declarations
struct parameters
{
    double m;       // Mass
    double C;       // Drag Coefficient
    double rho;     // Air density
    double eta;     // Viscosity
    double A;       // Surface area
    double h;       // Rider height
    double P;       // Power
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
    xarray<double> t = zeros<double>({100}); // Time (x)array
    xarray<double> X = zeros<double>({100}); // Data (x)array

    initialize(params, t, X);
    calculate(params, t, X);
    store(params, t, X);
}

void initialize(parameters &params, xarray<double> &t, xarray<double> &X)
{
    cout << "Running 'exercise2_3.exe'...\n\n";
    cout << "Input Parameters\n";
    cout << "================\n";
    cout << "m = ";
    cin >> params.m;
    cout << "C = ";
    cin >> params.C;
    cout << "rho = ";
    cin >> params.rho;
    cout << "eta = ";
    cin >> params.eta;
    cout << "A = ";
    cin >> params.A;
    cout << "P = ";
    cin >> params.P;
    cout << "dt = ";
    cin >> params.dt;
    cout << "t_f = ";
    cin >> params.tf;

    params.h = params.A / w_shoulders; // Assume rider's outline is a rectangle
    params.N = (unsigned int)ceil(params.tf / params.dt);

    // Resize our data (x)arrays to the correct size
    X.resize({params.N + 1});
    X.fill(0.);
    t.resize({params.N + 1});
    t.fill(0.);

    cout << "v0 = ";
    cin >> X(0);
    cout << "Done.\n\n";
    return;
}

void calculate(parameters &params, xarray<double> &t, xarray<double> &X)
{
    double F_appl; // Applied force on the bike
    double F_d;    // Drag force
    // Calculate the aggregate drag coefficients only once in advance
    double b1 = params.eta * params.A / params.h;
    double b2 = 0.5 * params.C * params.rho * params.A;
    cout << "Running Simulation...\n";

    for (unsigned int i = 0; i <= params.N - 1; i++)
    {
        // Forward Euler Method
        F_appl = params.P / X(i);
        F_d = -b1 * X(i) - b2 * fabs(X(i)) * X(i);
        X(i + 1) = X(i) + (1 / params.m) * (F_appl + F_d) * params.dt;
        t(i + 1) = t(i) + params.dt;
    }
    cout << "Done.\n\n";
    return;
}

void store(parameters &params, xarray<double> &t, xarray<double> &X)
{
    // Declare and open the output fine, then write data headers
    cout << "Writing to file...\n\n";
    ofstream f("../data/exercise2_3_output.csv");
    f << "m, C, rho, eta, A, P, dt, t_final, v0\n";
    f << params.m << ", " << params.C << ", " << params.rho << ", " << params.eta << ", " << params.A << ", " << params.P << ", " << params.dt << ", " << params.tf << ", " << X(0) << "\n";
    f << "t, v\n";

    // Write data to output file
    for (unsigned int i = 0; i <= params.N; i++)
    {
        f << t(i) << ", " << X(i) << "\n";
    }
    cout << "Done.\n\n";
    return;
}

// In Air: 70 0.5 1.225 0.00002 0.33 400 0.1 200
// In Water: 70 0.5 1000 0.003 0.33 400 0.1 200
