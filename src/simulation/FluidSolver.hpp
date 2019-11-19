#ifndef FLUID_SOLVER_HPP
#define FLUID_SOLVER_HPP

#include <glm/glm.hpp>
#include <vector>
#include <iostream>

class FluidSolver {
private:
    bool is_running_ = true;

public:
    FluidSolver(){};

    bool is_running() { return is_running_; };
    void terminate() { is_running_ = false; };

    void simulation();
    std::vector<glm::vec3> get_partical_position();
};

#endif