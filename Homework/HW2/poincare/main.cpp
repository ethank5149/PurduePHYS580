///\author Ethan Knox
///\date 10/6/2020.

#ifndef PURDUE_PHYS_580_POINCARE
#define PURDUE_PHYS_580_POINCARE
#define _USE_MATH_DEFINES
#endif //PURDUE_PHYS_580_POINCARE

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>
#include <functional>

#include <boost/program_options.hpp>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace std::placeholders;
namespace opt = boost::program_options;

typedef std::vector<double> state_t;
typedef boost::numeric::odeint::runge_kutta4<state_t> rk4;

int main(int argc, const char* argv[])
{
    double A, x_0, dx_0, Q, wd, sec_freq;
    long n_pnts, n_wait, pnt_density;

    opt::options_description params("Simulation Parameters");
    params.add_options()
            ("help,h", "Show Usage")
            (     "n_wait",opt::value<long>  (     &n_wait)->default_value(    100), "Number of Cycles Ignored For Transients To Disappear")
            (     "n_pnts",opt::value<long>  (     &n_pnts)->default_value(   10000), "Total Number of Cycles")
            ("pnt_density",opt::value<long>  (&pnt_density)->default_value(   1000), "Solution Accuracy (dt)")
            (          "A",opt::value<double>(          &A)->default_value(    1.35), "Amplitude")
            (        "x_0",opt::value<double>(        &x_0)->default_value(    0.2), "Initial Angle")
            (        "v_0",opt::value<double>(       &dx_0)->default_value(    0.0), "Initial Angular Frequency")
            (          "Q",opt::value<double>(          &Q)->default_value(    0.5), "Quality Factor")
            (        "w_d",opt::value<double>(         &wd)->default_value(2.0/3.0), "Driving Frequency")
            (   "sec_freq",opt::value<double>(   &sec_freq)->default_value(2.0/3.0), "Sectioning Frequency");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, params), vm);

    // Output help message if requested from the commandline
    if (vm.count("help")) {
        std::cout << params << std::endl;
        return 1;
    } // Necessary to retrieve the values from 'program_options'
    else {
        opt::notify(vm);
    }

    // Constants
    const long double dt = 2.0 * M_PI / (sec_freq * pnt_density);
    const long double t_i = 2.0 * M_PI / sec_freq;
    const long double t_f = (2.0 * M_PI * n_pnts) / sec_freq;
    const long double N = (n_pnts - 1) * pnt_density;

    // Initial Condition
    state_t x{x_0, dx_0};
    state_t dx_dt(2);

    // Data Series
    state_t x_series;
    state_t dx_series;
    state_t t_series;
    x_series.reserve(N);
    dx_series.reserve(N);
    t_series.reserve(N);

    // Data Recorder
    auto observer = [&](const state_t& x, const double t) {
        x_series.push_back(x.front());
        dx_series.push_back(x.back());
        t_series.push_back(t);
    };

    cout << "Running Simulation...\n";
    ofstream file("../data/poincare.dat");
    file << "# t \\theta \\frac{d\\theta}{dt}\n";
    file << setprecision(numeric_limits<long double>::digits10 + 1);

    // ODE System
    std::function<void(const state_t&, state_t&, double)> sys = [&](const state_t& x, state_t& dx_dt, const double t) {
        dx_dt.front() = x.back();
        dx_dt.back() = -sin(x.front()) - Q * x.back() + A * sin(wd * t);
    };

    // Integrate
    boost::numeric::odeint::integrate_n_steps(rk4(), sys, x, t_i, dt, N, observer);

    // Restrict the domain to [-pi, pi]
    for(auto &theta : x_series){
        theta = fmod(theta + M_PI, 2.0 * M_PI) - M_PI;
    }

    // Write data to file
    for(size_t i=0; i<x_series.size(); i++){
        if((i > pnt_density) && (i / pnt_density > n_wait) && (i % pnt_density == 0)){
            file << std::scientific << t_series.at(i) << " " << x_series.at(i) << " " << dx_series.at(i) << "\n";
        }
    }

    cout << "Done.\n\n";
    return 0;
}