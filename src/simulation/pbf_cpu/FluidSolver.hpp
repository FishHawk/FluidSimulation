#ifndef FLUID_SOLVER_PBF_CPU_HPP
#define FLUID_SOLVER_PBF_CPU_HPP

#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <vector>

#include "../FluidSolver.hpp"
#include "Particles.hpp"

namespace Simulation {
namespace PbfCpu {

class FluidSolver : public ::Simulation::FluidSolver {
private:
    Particles particles_;
    double particle_radius_ = 0.025;
    size_t fluid_particals_number_ = 0;
    size_t boundary_particals_number_ = 0;
    double sph_radius_ = 4 * 0.025;
    double fluid_density_ = 1000.0;

    std::chrono::system_clock::time_point time_point_;
    float time_step_ = 0.005;

    void update_time_step();

    void reset_acceleration();
    void update_particles_ignore_constraint(float delta_time);
    void constraint_projection();
    void correct_velocity(float delta_time);

    std::vector<double> calculate_fluid_density(std::map<glm::ivec3, std::vector<int>> &neighbors);
    std::vector<double> calculate_lagrange_multiplier(std::vector<double> &densities, std::map<glm::ivec3, std::vector<int>> &neighbors);
    void solve_constraint(std::vector<double> &lambdas, std::map<glm::ivec3, std::vector<int>> &neighbors);

    FluidSolver() = default;
    FluidSolver(FluidSolver const &) = delete;
    void operator=(FluidSolver const &) = delete;

public:
    static FluidSolver &get_instance() {
        static FluidSolver instance;
        return instance;
    };
    ~FluidSolver() = default;

    // config
    void set_particle_radius(double radius) {
        particle_radius_ = radius;
        sph_radius_ = 4 * radius;
    }
    void setup_model(const std::vector<glm::vec3> &fluid_particles,
                     const std::vector<glm::vec3> &boundary_particles);

    void simulate() override;
    std::vector<glm::vec3> get_partical_position() override;
};

}  // namespace PbfCpu
}  // namespace Simulation

#endif