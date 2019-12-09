#ifndef SIMULATE_CPU_SIMULATE_SYSTEM_HPP
#define SIMULATE_CPU_SIMULATE_SYSTEM_HPP

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "simulate/SimulateSystem.hpp"
#include "Particles.hpp"

namespace simulate {
namespace cpu {

class SimulateSystem : public ::simulate::SimulateSystem {
private:
    Particles particles_;
    double particle_radius_ = 0.025;
    size_t fluid_particles_number_ = 0;
    size_t boundary_particles_number_ = 0;
    double sph_radius_ = 4 * 0.025;
    double fluid_density_ = 1000.0;

    std::chrono::system_clock::time_point time_point_;
    float time_step_ = 0.005;

    void update_time_step();

    void reset_acceleration();
    void calculate_predicted_positions(float delta_time);
    void update_particles(float delta_time);

    std::vector<double> calculate_lagrange_multiplier(std::unordered_map<glm::ivec3, std::vector<int>> &neighbors);
    void solve_constraint(std::vector<double> &lambdas, std::unordered_map<glm::ivec3, std::vector<int>> &neighbors);

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
        particle_radius_ = radius;
        sph_radius_ = 4 * radius;
    }

    
    std::vector<glm::vec3> get_particle_position() override;
};

} // namespace cpu
} // namespace simulate

#endif