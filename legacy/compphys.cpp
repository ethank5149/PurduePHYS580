#include "compphys.hpp"

int main()
{
    struct Simulation sim;

    sim.obj.vel = {4.0, 0.0, 0.0};
    sim.dt = 0.001;

    sim.init_memory();
    sim.run();
    sim.store();
    sim.obj.print_current_state();
}