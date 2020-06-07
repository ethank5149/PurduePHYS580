/*---------------------------------------------------------------------------*\
  ========          |  Purdue Physics 580 - Computational Physics
  \\                |  Chapter 1 - Exercise 1
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
/*****************************************************************************/

// Using
/*****************************************************************************/
using namespace std;
using namespace xt;

// Global Definitions
#define v0 40 //Horizontal Velocity [m/s]

// Function Declarations
/*****************************************************************************/
/*(dt, t, X, dX, N)*/
void initialize(double &, xarray<double> &, double &, xarray<double> &, xarray<double> &, unsigned int &);
void calculate(double, xarray<double> &, double, xarray<double> &, xarray<double> &, unsigned int &);
void store(double, xarray<double> &, double, xarray<double> &, xarray<double> &, unsigned int);
/*****************************************************************************/

int main()
{
    xarray<double> t = zeros<double>({100});     // Time (x)array
    xarray<double> X = zeros<double>({100, 2});  // x, y Data (x)array
    xarray<double> dX = zeros<double>({100, 2}); // dx, dy Data (x)array

    double dt; //time step
    double t_final;
    unsigned int N; //Number of datapoints

    initialize(dt, t, t_final, X, dX, N);
    calculate(dt, t, t_final, X, dX, N);
    store(dt, t, t_final, X, dX, N);
}

void initialize(double &dt, xarray<double> &t, double &t_final, xarray<double> &X, xarray<double> &dX, unsigned int &N)
{
    // Initialize temporary variable for inputting initial conditions
    double x_0;
    double y_0;

    cout << "Initializing Simulation...\n";
    cout << "Input Initial Conditions & Parameters:\n";
    cout << "Format: dt t_final x_0 y_0\n";
    cin >> dt >> t_final >> x_0 >> y_0;
    N = (unsigned int)ceil(t_final / dt);

    // Construct our data (x)arrays
    X.resize({N + 1, 2});
    X.fill(0.);
    dX.resize({N + 1, 2});
    dX.fill(0.);
    t.resize({N + 1});
    t.fill(0.);

    // Assign the initial conditions to their respective vectors
    row(X, 0) = xarray<double>{x_0, y_0};
    // Since the velocity is constant
    view(dX, all(), 0).fill(v0);
    cout << "Done.\n\n";

    cout << "Initialized State\n";
    cout << "##################################################\n";
    cout << "Number of timesteps: N = ceiling(" << t_final << "/" << dt << ") = " << N << "\n";
    cout << "Therefore: t = (0, N*dt) = (0.0, " << N * dt << ")\n";
    cout << "Initial Position = <" << x_0 << ", " << y_0 << ">\n";
    cout << "##################################################\n\n";
    return;
}

void calculate(double dt, xarray<double> &t, double t_final, xarray<double> &X, xarray<double> &dX, unsigned int &N)
{
    unsigned int i = 0;
    cout << "Running Simulation...\n";

    for (i = 0; i <= N - 1; i++)
    {
        //Forward Euler Method
        row(X, i + 1) = row(X, i) + row(dX, i) * dt;
        t(i + 1) = t(i) + dt;
    }
    cout << "Done.\n\n";
    return;
}

void store(double dt, xarray<double> &t, double t_final, xarray<double> &X, xarray<double> &dX, unsigned int N)
{
    // Define output file string
    stringstream ss;
    ss << "../data/exercise1_2_dt=" << dt << "_tf=" << t_final << "_x0=" << X(0, 0) << "_y0=" << X(0, 1) << ".csv";

    // Declare and open the output fine, then write data headers
    cout << "Writing to file...\n\n";
    ofstream f(ss.str().c_str());
    f << "t, x, y, dx, dy\n";

    // Write data to output file
    for (int i = 0; i <= N; i++)
    {
        f << t(i) << ", " << X(i, 0) << ", " << X(i, 1) << ", " << dX(i, 0) << ", " << dX(i, 1) << "\n";
    }
    cout << "Done.\n\n";
    return;
}
