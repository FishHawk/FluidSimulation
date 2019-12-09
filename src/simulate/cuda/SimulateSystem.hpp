#ifndef SIMULATE_CUDA_SIMULATE_SYSTEM_HPP
#define SIMULATE_CUDA_SIMULATE_SYSTEM_HPP

#include <chrono>
#include <iostream>
#include <map>
#include <vector>

#include <glm/glm.hpp>

#include "simulate/SimulateSystem.hpp"
#include "FluidSolver.cuh"

namespace simulate {
namespace cuda {

class SimulateSystem : public ::simulate::SimulateSystem {
private:
    std::vector<glm::vec4> positions_;
    std::vector<glm::vec4> initial_positions_;
    size_t fluid_particles_number_ = 0;
    size_t boundary_particles_number_ = 0;

    std::chrono::system_clock::time_point time_point_;
    float time_step_ = 0.016;

    SimulateParams m_params;
    FluidSolverCuda solver_;

    SimulateSystem() = default;
    SimulateSystem(SimulateSystem const &) = delete;
    void operator=(SimulateSystem const &) = delete;

public:
    static SimulateSystem &get_instance() {
        static SimulateSystem instance;
        return instance;
    };
    ~SimulateSystem() = default;

    // simulate
    void simulate() override;
    void reset() override;

    // fluid particles
    void set_particles_position(const std::vector<glm::vec3> &particles_initial_positions) override;
    void set_particles_radius(double radius) {
        // particle_radius_ = radius;
        // sph_radius_ = 4 * radius;
    }

    std::vector<glm::vec3> get_particle_position() override;
};

} // namespace cuda
} // namespace simulate

#endif