#pragma once
// compphys.hpp

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <valarray>

using namespace std;

struct Forces
{
    valarray<double> gravity;
    valarray<double> applied;
    valarray<double> viscous_drag;
    valarray<double> laminar_drag;
    vector<valarray<double> (*)(struct Simulation)> force_list;
};

struct Environment
{
    double rho = 1.225;   // Default in air
    double eta = 0.00002; // Default in air
    double T = 0.0;       // Not implemented
    double P = 400.0;     // Average power output
    double theta = 0.0;   // Incline Angle
    double g = 9.81;      //Gravitational Acceleration [m/s^2]
};

struct Data
{
    vector<valarray<double>> position;
    vector<valarray<double>> velocity;
};

struct Object
{
    double m = 70.0;                          // Mass [kg]
    double Cd = 0.5;                          // Drag Coefficient
    double A = 0.33;                          // Cross-sectional Area [m^2]
    double I = 0.0;                           // Moment of Inertia
    double w = 0.411;                         // Width
    double h = A / w;                         // Height
    valarray<double> pos = {0.0, 0.0, 0.0};   // Position
    valarray<double> vel = {0.0, 0.0, 0.0};   // Velocity
    valarray<double> F_Net = {0.0, 0.0, 0.0}; // Net Force
    void print_current_state();
};

void Object::print_current_state()
{
    cout << "Position, Velocity, Net Force\n";
    cout << "<" << pos[0] << ", " << pos[1] << ", " << pos[2] << ">, ";
    cout << "<" << vel[0] << ", " << vel[1] << ", " << vel[2] << ">, ";
    cout << "<" << F_Net[0] << ", " << F_Net[1] << ", " << F_Net[2] << ">\n";
}

struct Simulation
{

    struct Environment env;                    // Simulation Environment
    struct Object obj;                         // Simulation Object
    struct Data timeseries;                    // Simulation Time-series Data
    struct Forces forces;                      // Simulation Forces
    double dt = 0.1;                           // Time-step
    double tf = 200;                           // Final Time
    const unsigned int N = (int)(tf / dt) + 1; // Number of Simulation Update Steps
    void init_memory();
    void update_step();
    void log();
    void run();
    void store();
    bool terminate();
};

void Simulation::init_memory()
{
    //N = (int)(tf / dt) + 1;
    // Allocate Space for Enough Data
    timeseries.position.reserve(N);
    timeseries.velocity.reserve(N);
}

void Simulation::update_step()
{
    // Calculate All Forces

    // forces.gravity = {-obj.m * env.g * sin(env.theta), 0, 0};
    // forces.applied = env.P * obj.vel / pow(obj.vel, 2.0).sum();
    // forces.viscous_drag = -(env.eta * obj.A / obj.h) * obj.vel;
    // forces.laminar_drag = -0.5 * obj.Cd * env.rho * obj.A * pow(pow(obj.vel, 2.0).sum(), 0.5) * obj.vel;
    // obj.F_Net = forces.laminar_drag + forces.viscous_drag + forces.applied + forces.gravity;

    obj.F_Net = {0.0, 0.0, 0.0};
    for (auto &force : forces.force_list)
    {
        obj.F_Net += force(*this);
    }

    // Can adapt different numerical methods here
    obj.vel += dt * obj.F_Net / obj.m;
    obj.pos += dt * obj.vel;
}

void Simulation::log()
{
    // Append Current State to Time-series Data
    timeseries.position.push_back(obj.pos);
    timeseries.velocity.push_back(obj.vel);
}

bool Simulation::terminate()
{
    return false;
}

void Simulation::run()
{
    for (unsigned int i = 0; i < N; i++)
    {
        log();
        update_step();
        if (terminate())
        {
            // Free Unused Memory Previously Reserved
            timeseries.position.shrink_to_fit();
            timeseries.velocity.shrink_to_fit();
            break;
        }
    }
}

void Simulation::store()
{
    // Define output file string
    stringstream ss;
    ss << "data/output.csv";

    // Declare and open the output fine, then write data headers
    ofstream f(ss.str().c_str());
    f << "t, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z\n";

    // Write data to output file
    for (unsigned int i = 0; i < N; i++)
    {
        valarray<double> p = timeseries.position[i];
        valarray<double> v = timeseries.velocity[i];

        f << i * dt << ", " << p[0] << ", " << p[1] << ", " << p[2] << ", ";
        f << v[0] << ", " << v[1] << ", " << v[2] << "\n";
    }
    return;
}