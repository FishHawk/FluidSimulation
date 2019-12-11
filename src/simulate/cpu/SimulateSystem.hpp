#ifndef SIMULATE_CPU_SIMULATE_SYSTEM_HPP
#define SIMULATE_CPU_SIMULATE_SYSTEM_HPP

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "Particles.hpp"
#include "simulate/SimulateSystem.hpp"

namespace simulate {
namespace cpu {

class SimulateSystem : public ::simulate::SimulateSystem {
private:
    Particles particles_;
    size_t fluid_particles_number_ = 0;

    float density_{1.0f / (8.0f * powf(particle_radius_, 3.0f))};
    float inv_density_{8.0f * powf(particle_radius_, 3.0f)};

    void calculate_predicted_positions(float delta_time);
    std::vector<float> calculate_lagrange_multiplier(std::unordered_map<glm::ivec3, std::vector<int>> &neighbors);
    void solve_constraint(std::vector<float> &lambdas, std::unordered_map<glm::ivec3, std::vector<int>> &neighbors);
    void update_particles(float delta_time);

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
    void set_particle_position(const std::vector<glm::vec3> &particles_initial_positions) override;

    void apply() override;
    std::vector<glm::vec3> get_particle_position() override;
};

} // namespace cpu
} // namespace simulate

#endif