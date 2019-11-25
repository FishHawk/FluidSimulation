#ifndef FLUID_SOLVER_PBF_CUDA_HPP
#define FLUID_SOLVER_PBF_CUDA_HPP

#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <vector>

#include "../FluidSolver.hpp"

namespace Simulation {
namespace PbfCuda {

class FluidSolver : public ::Simulation::FluidSolver {
private:
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
        // particle_radius_ = radius;
        // sph_radius_ = 4 * radius;
    }
    void setup_model(const std::vector<glm::vec3> &fluid_particles,
                     const std::vector<glm::vec3> &boundary_particles);

    void simulate() override;
    std::vector<glm::vec3> get_partical_position() override;
};

}  // namespace PbfCuda
}  // namespace Simulation

#endif