#ifndef FLUID_SOLVER_HPP
#define FLUID_SOLVER_HPP

#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

#include "Particles.hpp"

class FluidSolver {
private:
    bool is_running_ = true;

    Particles particles_;
    double particle_radius_ = 0.025;
    double fluid_density_ = 1000.0;

    std::chrono::system_clock::time_point time_point_;
    float time_step_ = 0.005;

    void update_time_step();

    void reset_acceleration();
    void update_particles_ignore_constraint(float delta_time);
    void correct_velocity(float delta_time);

public:
    FluidSolver();

    bool is_running() { return is_running_; };
    void terminate() { is_running_ = false; };

    void setup_model(const std::vector<glm::vec3> &fluid_particles,
                     const std::vector<glm::vec3> &boundary_particles);

    void simulation();

    std::vector<glm::vec3> get_partical_position();
};

#endif