#include "compphys.hpp"

valarray<double> F_gravity(struct Simulation sim)
{
    return {-sim.obj.m * sim.env.g * sin(sim.env.theta), 0, 0};
}

valarray<double> F_applied(struct Simulation sim)
{
    return sim.env.P * sim.obj.vel / pow(sim.obj.vel, 2.0).sum();
}

valarray<double> F_viscous_drag(struct Simulation sim)
{
    return -(sim.env.eta * sim.obj.A / sim.obj.h) * sim.obj.vel;
}

valarray<double> F_laminar_drag(struct Simulation sim)
{
    return -0.5 * sim.obj.Cd * sim.env.rho * sim.obj.A * pow(pow(sim.obj.vel, 2.0).sum(), 0.5) * sim.obj.vel;
}

valarray<double> F_Net(struct Simulation sim)
{
    return F_applied(sim) + F_gravity(sim) + F_viscous_drag(sim) + F_laminar_drag(sim);
}

int main()
{
    valarray<double> (*F_Net_ptr)(struct Simulation) = F_Net;

    struct Simulation sim;

    sim.obj.F_Net = F_Net_ptr;
    sim.obj.vel = {4.0, 0.0, 0.0};
    sim.dt = 0.01;

    sim.init_memory();
    sim.run();
    sim.store();
    sim.print_current_state();
}