/*---------------------------------------------------------------------------*\
  ========          |  Purdue Physics 580 - Computational Physics
  \\                |  Chapter 2 - Exercise 1
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
    xarray<double> t = zeros<double>({100});           // Time (x)array
    xarray<double> X = zeros<double>({100, num_eqns}); // Data (x)array

    initialize(params, t, X);
    calculate(params, t, X);
    store(params, t, X);
}

void initialize(parameters &params, xarray<double> &t, xarray<double> &X)
{
    cout << "Running 'exercise2_1.exe'...\n\n";
    cout << "Input Parameters (m C rho A P v0 dt t_final):\n";
    cin >> params.m >> params.C >> params.rho >> params.A >> params.P >> params.v0 >> params.dt >> params.tf;

    params.N = (unsigned int)ceil(params.tf / params.dt);
    // Resize our data (x)arrays to the correct size
    X.resize({params.N + 1, num_eqns});
    X.fill(0.);
    t.resize({params.N + 1});
    t.fill(0.);

    cout << "Input Initial Conditions (x_0 y_0):\n";
    cin >> X(0, 0) >> X(0, 1);
    cout << "Done.\n\n";

    // Print the parsed input back to the user
    stringstream input_params_string;
    stringstream input_initcond_string;
    input_params_string << "(" << params.m << ", " << params.C << ", " << params.rho << ", " << params.A << ", " << params.P << ", " << params.v0 << ", " << params.dt << ", " << params.tf << ")\n";
    input_initcond_string << "(" << X(0, 0) << ", " << X(0, 1) << ")\n";

    cout << "Parsed Input\n";
    cout << "##################################################\n";
    cout << "Parameters (m C rho A P v0 dt t_final) = " << input_params_string.str();
    cout << "Initial Conditions (x_0, y_0) = " << input_initcond_string.str();
    cout << "##################################################\n\n";
    return;
}

void calculate(parameters &params, xarray<double> &t, xarray<double> &X)
{
    xarray<double> F_appl{params.P / params.v0, 0};    // Applied force on the bike
    xarray<double> F_d;                                // Drag force
    double b = 0.5 * params.C * params.rho * params.A; // Calculate the aggregate drag coefficient only once in advance
    cout << "Running Simulation...\n";

    for (unsigned int i = 0; i <= params.N - 1; i++)
    {
        // Forward Euler Method
        F_d = -b * norm_l2(row(X, i)) * row(X, i);
        row(X, i + 1) = row(X, i) + (1 / params.m) * (F_appl + F_d) * params.dt;
        t(i + 1) = t(i) + params.dt;
    }
    cout << "Done.\n\n";
    return;
}

void store(parameters &params, xarray<double> &t, xarray<double> &X)
{
    // Define output file string
    stringstream ss;
    ss << "../data/exercise2_1_m=" << params.m << "_C=" << params.C << "_r=" << params.rho << "_A=" << params.A << "_P=" << params.P << "_v0=" << params.v0 << "_dt=" << params.dt << "_tf=" << params.tf << "_x0=" << X(0, 0) << "_y0=" << X(0, 1) << ".csv";

    // Declare and open the output fine, then write data headers
    cout << "Writing to file...\n\n";
    ofstream f(ss.str().c_str());
    f << "t, x, y\n";

    // Write data to output file
    for (unsigned int i = 0; i <= params.N; i++)
    {
        f << t(i) << ", " << X(i, 0) << ", " << X(i, 1) << "\n";
    }
    cout << "Done.\n\n";
    return;
}
// 70 0.5 1.1445 0.33 400 4 0.1 200