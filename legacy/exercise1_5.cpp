/*---------------------------------------------------------------------------*\
  ========          |  Purdue Physics 580 - Computational Physics
  \\                |  Chapter 1 - Exercise 5
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
/*****************************************************************************/

// Global Definitions
#define g 9.81 //Gravitational Acceleration [m/s^2]

// Function Declarations
/*****************************************************************************/
/*(dt, t, X, dX, N)*/
void initialize(double &, double &, double &, xarray<double> &, double &, xarray<double> &, xarray<double> &, unsigned int &);
void calculate(double &, double &, double &, xarray<double> &, double &, xarray<double> &, xarray<double> &, unsigned int &);
void store(double &, double &, double &, xarray<double> &, double &, xarray<double> &, xarray<double> &, unsigned int &);

int main()
{
    xarray<double> t = zeros<double>({100});     // Time (x)array
    xarray<double> X = zeros<double>({100, 2});  // Amount (x)array
    xarray<double> dX = zeros<double>({100, 2}); // Derivative (x)array

    double t_A; //Half-Life of A
    double t_B; //Half-Life of B
    double dt;  //time step
    double t_final;
    unsigned int N; //Number of datapoints

    initialize(t_A, t_B, dt, t, t_final, X, dX, N);
    calculate(t_A, t_B, dt, t, t_final, X, dX, N);
    store(t_A, t_B, dt, t, t_final, X, dX, N);
}

void initialize(double &t_A, double &t_B, double &dt, xarray<double> &t, double &t_final, xarray<double> &X, xarray<double> &dX, unsigned int &N)
{
    // Initialize temporary variable for inputting initial conditions
    double A_0;
    double B_0;

    cout << "Initializing Simulation...\n";
    cout << "Input Initial Conditions & Parameters:\n";
    cout << "Format: t_A t_B dt t_final A_0 B_0\n";
    cin >> t_A >> t_B >> dt >> t_final >> A_0 >> B_0;
    N = (unsigned int)ceil(t_final / dt);

    // Construct our data (x)arrays
    X.resize({N + 1, 2});
    X.fill(0.);
    dX.resize({N + 1, 2});
    dX.fill(0.);
    t.resize({N + 1});
    t.fill(0.);

    // Assign the initial conditions to their respective vectors
    row(X, 0) = xarray<double>{A_0, B_0};
    dX(0, 0) = X(0, 1) / t_B - X(0, 0) / t_A;
    dX(0, 1) = X(0, 0) / t_A - X(0, 1) / t_B;
    cout << "Done.\n\n";

    cout << "Initialized State\n";
    cout << "##################################################\n";
    cout << "Number of timesteps: N = ceiling(" << t_final << "/" << dt << ") = " << N << "\n";
    cout << "Therefore: t = (0, N*dt) = (0.0, " << N * dt << ")\n";
    cout << "Initial Amounts = <" << A_0 << ", " << B_0 << ">\n";
    cout << "Rate Constants = <" << t_A << ", " << t_B << ">\n";
    cout << "##################################################\n\n";
    return;
}

void calculate(double &t_A, double &t_B, double &dt, xarray<double> &t, double &t_final, xarray<double> &X, xarray<double> &dX, unsigned int &N)
{
    unsigned int i = 0;

    cout << "Running Simulation...\n";
    for (i = 0; i <= N - 1; i++)
    {
        //Forward Euler Method
        dX(i + 1, 0) = X(i, 1) / t_B - X(i, 0) / t_A;
        dX(i + 1, 1) = X(i, 0) / t_A - X(i, 1) / t_B;
        row(X, i + 1) = row(X, i) + row(dX, i) * dt;
        t(i + 1) = t(i) + dt;
    }
    cout << "Done.\n\n";
    return;
}

void store(double &t_A, double &t_B, double &dt, xarray<double> &t, double &t_final, xarray<double> &X, xarray<double> &dX, unsigned int &N)
{
    // Define output file string
    stringstream ss;
    ss << "../data/exercise1_5_tA=" << t_A << "_tB=" << t_B << "_dt=" << dt << "_tf=" << t_final << "_A0=" << X(0, 0) << "_B0=" << X(0, 1) << ".csv";

    // Declare and open the output fine, then write data headers
    cout << "Writing to file...\n\n";
    ofstream f(ss.str().c_str());
    f << "t, A, B, dA, dB\n";

    // Write data to output file
    for (int i = 0; i <= N; i++)
    {
        f << t(i) << ", " << X(i, 0) << ", " << X(i, 1) << ", " << dX(i, 0) << ", " << dX(i, 1) << "\n";
    }
    cout << "Done.\n\n";
    return;
}
