#ifndef SIMULATE_CUDA_SIMULATE_SYSTEM_HPP
#define SIMULATE_CUDA_SIMULATE_SYSTEM_HPP

#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <vector>

#include "../SimulateSystem.hpp"
#include "FluidSolver.cuh"

namespace simulate {
namespace cuda {

class SimulateSystem : public ::simulate::SimulateSystem {
private:
    std::vector<glm::vec4> positions_;
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

} // namespace cuda
} // namespace simulate

#endif