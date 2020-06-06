// #include "compphys.hpp"
// valarray<double> Fappl(double P, valarray<double> vel)
// {
//     valarray<double> F_appl = P * vel / pow(vel, 2.0).sum();
//     return F_appl;
// }
// int main()
// {
//     struct Simulation sim;
//     // sim.particle.Cd = 0.0;
//     sim.obj.vel = {4.0, 0.0, 0.0};
//     valarray<double> (*F_appl_ptr)(double, valarray<double>) = Fappl;
//     sim.forces.force_list.push_back(F_appl_ptr);
//     sim.init_memory();
//     sim.run();
//     sim.store();
//     sim.obj.print_current_state();
// }

#include "compphys.hpp"

valarray<double> F_gravity(struct Simulation sim)
{
    valarray<double> F_gravity = {-sim.obj.m * sim.env.g * sin(sim.env.theta), 0, 0};
    return F_gravity;
}

valarray<double> F_applied(struct Simulation sim)
{
    valarray<double> F_applied = sim.env.P * sim.obj.vel / pow(sim.obj.vel, 2.0).sum();
    return F_applied;
}

valarray<double> F_viscous_drag(struct Simulation sim)
{
    valarray<double> F_viscous_drag = -(sim.env.eta * sim.obj.A / sim.obj.h) * sim.obj.vel;
    return F_viscous_drag;
}

valarray<double> F_laminar_drag(struct Simulation sim)
{
    valarray<double> F_laminar_drag = -0.5 * sim.obj.Cd * sim.env.rho * sim.obj.A * pow(pow(sim.obj.vel, 2.0).sum(), 0.5) * sim.obj.vel;
    return F_laminar_drag;
}

int main()
{
    valarray<double> (*F_applied_ptr)(struct Simulation) = F_applied;
    valarray<double> (*F_gravity_ptr)(struct Simulation) = F_gravity;
    valarray<double> (*F_laminar_drag_ptr)(struct Simulation) = F_laminar_drag;
    valarray<double> (*F_viscous_drag_ptr)(struct Simulation) = F_viscous_drag;

    struct Simulation sim;

    sim.forces.force_list.push_back(F_applied_ptr);
    sim.forces.force_list.push_back(F_gravity_ptr);
    // sim.forces.force_list.push_back(F_viscous_drag_ptr);
    // sim.forces.force_list.push_back(F_laminar_drag_ptr);
    sim.obj.vel = {4.0, 0.0, 0.0};

    sim.init_memory();
    sim.run();
    sim.store();
    sim.obj.print_current_state();
}