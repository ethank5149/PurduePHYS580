///\author Ethan Knox
///\date 8/2/2020.

#ifndef PURDUE_PHYS_580_BIFURCATION
#define PURDUE_PHYS_580_BIFURCATION
#define _USE_MATH_DEFINES
#endif //PURDUE_PHYS_580_BIFURCATION

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
    double A_i, A_f, x_0, dx_0, Q, wd, phase_cut, sec_freq;
    long n_A, n_pnts, n_wait, pnt_density;

    opt::options_description params("Simulation Parameters");
    params.add_options()
            ("help,h", "Show Usage")
            (        "n_A",opt::value<long>  (        &n_A)->default_value(   1000), "Number of Amplitudes")
            (     "n_wait",opt::value<long>  (     &n_wait)->default_value(    300), "Number of Cycles Ignored For Transients To Disappear")
            (     "n_pnts",opt::value<long>  (     &n_pnts)->default_value(   1000), "Number of Points")
            ("pnt_density",opt::value<long>  (&pnt_density)->default_value(    100), "Solution Accuracy (dt)")
            (        "A_i",opt::value<double>(        &A_i)->default_value(   1.35), "Initial Amplitude")
            (        "A_f",opt::value<double>(        &A_f)->default_value(    1.5), "Final Amplitude")
            (        "x_0",opt::value<double>(        &x_0)->default_value(    0.2), "Initial Angle")
            (        "v_0",opt::value<double>(       &dx_0)->default_value(    0.0), "Initial Angular Frequency")
            (          "Q",opt::value<double>(          &Q)->default_value(    0.5), "Quality Factor")
            (        "w_d",opt::value<double>(         &wd)->default_value(2.0/3.0), "Driving Frequency")
            (  "phase_cut",opt::value<double>(  &phase_cut)->default_value(    0.0), "Phase Cut")
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
    const long double dA = (A_f - A_i) / n_A;
    const long double dt = 2.0 * M_PI / (sec_freq * pnt_density);
    const long double t_i = (2.0 * M_PI + phase_cut) / sec_freq;
    const long double t_f = (2.0 * M_PI * n_pnts + phase_cut) / sec_freq;
    const long double N = (n_pnts - 1) * pnt_density;

    // Initial Condition
    state_t x{x_0, dx_0};
    state_t dx_dt(2);

    // Data Series
    std::vector<double> x_series;
    std::vector<double> dx_series;
    std::vector<double> t_series;

    // Data Recorder
    auto observer = [&](const state_t& x, const double t) {
        x_series.push_back(x[0]);
        dx_series.push_back(x[1]);
        t_series.push_back(t);
    };

    cout << "Running Simulation...\n";
    ofstream file("../data/bifurcation.dat");
    file << "# Amplitude X\n";
    file << setprecision(numeric_limits<long double>::digits10 + 1);

    long double A = A_i;
    for (unsigned int j = 0; j<n_A;j++){
        cout  << 100.0 * j / n_A << '%' << endl;

        // ODE System
        std::function<void(const state_t&, state_t&, double)> sys = [&](const state_t& x, state_t& dx_dt, const double t) {
            dx_dt[0] = x[1];
            dx_dt[1] = -sin(x[0]) - Q * x[1] + A * sin(wd * t);
        };

        // Integrate
        boost::numeric::odeint::integrate_n_steps(rk4(), sys, x, t_i, dt, N, observer);

        // Restrict the domain to 9-pi, pi]
        for(auto &theta : x_series){
            theta = fmod(theta + M_PI, 2.0 * M_PI) - M_PI;
        }

        // Write data to file
        for(size_t i=0; i<x_series.size(); i++){
            if(i%pnt_density==0){
                file << std::scientific << A << " " << x_series.back() << "\n";
            }
        }

        // Reset data containers
        x_series.clear();
        dx_series.clear();
        t_series.clear();
        A += dA;
    }
    cout << "Done.\n\n";
    return 0;
}