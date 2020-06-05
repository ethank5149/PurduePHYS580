// Matrix.cpp

#include <iostream>
#include <vector>
#include <valarray>

using namespace std;

#define g 9.81 //Gravitational Acceleration [m/s^2]

struct Data
{
    vector<valarray<double>> pos_timeseries;
    vector<valarray<double>> vel_timeseries;
    vector<valarray<double>> acc_timeseries;
    void print();
} sim_data;

void Data::print()
{
    cout << "Position, Velocity, Acceleration\n";
    for (int i = 0; i < pos_timeseries.size(); i++)
    {
        valarray<double> p = pos_timeseries[i];
        valarray<double> v = vel_timeseries[i];
        valarray<double> a = acc_timeseries[i];

        cout << "<" << p[0] << ", " << p[1] << ", " << p[2] << ">, ";
        cout << " ";
        cout << "<" << v[0] << ", " << v[1] << ", " << v[2] << ">, ";
        cout << " ";
        cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ">";
        cout << "\n";
    }
}

struct Parameters
{
    double dt;
    double tf;
    double N;
    double P = 400;     // Power
    double v0;          // Initial Velocity
    double rho = 1.225; // Air density

} params;

struct Object
{
    // Initialize to zeros
    valarray<double> pos = {0.0, 0.0, 0.0}; // Position
    valarray<double> vel = {4.0, 0.0, 0.0}; // Velocity
    valarray<double> acc = {0, 0, 0};       // Acceleration
    valarray<double> F_d = {0, 0, 0};       // Drag force
    valarray<double> F_appl = {0, 0, 0};    // Applied force
    valarray<double> F_Net = {0, 0, 0};     // Net force
    double m;                               // Mass
    double C = 0.50;                        // Drag Coefficient
    double A = 0.33;                        // Cross-sectional area
    void update_step();
} particle;

void Object::update_step()
{
    // Can adapt different numerical methods here
    F_d = -0.5 * C * params.rho * A * pow(pow(vel, 2.0).sum(), 0.5) * vel;
    F_appl = params.P * vel / pow(vel, 2.0).sum();

    F_Net = F_appl + F_d;
    acc = F_Net / m;

    // Log data now to include initial conditions
    sim_data.acc_timeseries.push_back(acc);
    sim_data.vel_timeseries.push_back(vel);
    sim_data.pos_timeseries.push_back(pos);

    vel += params.dt * acc;
    pos += params.dt * vel;
}

void init_data(Object &particle, Data &sim_data, Parameters &params)
{
    // cout << "x0 y0 z0 = ";
    // cin >> particle.pos[0] >> particle.pos[1] >> particle.pos[2];
    // cout << "vx0 vy0 vz0 = ";
    // cin >> particle.vel[0] >> particle.vel[1] >> particle.vel[2];
    params.v0 = pow(pow(particle.vel, 2.0).sum(), 0.5);
    cout << "m = ";
    cin >> particle.m;
    // cout << "A = ";
    // cin >> particle.A;
    // cout << "C = ";
    // cin >> particle.C;
    cout << "dt = ";
    cin >> params.dt;
    cout << "tf = ";
    cin >> params.tf;

    params.N = (int)(params.tf / params.dt);
    sim_data.pos_timeseries.reserve(params.N);
    sim_data.vel_timeseries.reserve(params.N);
    sim_data.acc_timeseries.reserve(params.N);
}

void run(Object &particle, Data &sim_data, Parameters &params)
{
    for (int i = 0; i < params.N; i++)
    {
        // Update our particle in time
        particle.update_step();
    }
}

int main()
{
    init_data(particle, sim_data, params);
    run(particle, sim_data, params);
    sim_data.print();
}