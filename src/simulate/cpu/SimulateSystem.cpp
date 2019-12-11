#include "SimulateSystem.hpp"

#include <glm/gtx/hash.hpp>

#include "SplineInterpolation.hpp"

// expand glm ivec3 for map
template <>
struct std::less<glm::ivec3> {
    bool operator()(const glm::ivec3 &lhs, const glm::ivec3 &rhs) const {
        return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
    }
};

// expand glm ivec3 for unordered map
template <>
struct std::equal_to<glm::ivec3> {
    bool operator()(const glm::ivec3 &lhs, const glm::ivec3 &rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

using namespace simulate::cpu;

void SimulateSystem::set_particle_position(const std::vector<glm::vec3> &particles_initial_positions) {
    fluid_particles_number_ = particles_initial_positions.size();

    particles_.clear();
    particles_.reserve(fluid_particles_number_);
    for (const auto &position : particles_initial_positions) {
        particles_.add(position);
    }
}

void SimulateSystem::reset() {
    stop();
    std::lock_guard<std::mutex> lock(m);
    particles_.reset();
}

void SimulateSystem::apply() {
    density_ = 1.0f / (8.0f * powf(particle_radius_, 3.0f));
    inv_density_ = 8.0f * powf(particle_radius_, 3.0f);
}

std::vector<glm::vec3> SimulateSystem::get_particle_position() {
    std::vector<glm::vec3> fluid_particle_positions;
    for (int i = 0; i < fluid_particles_number_; i++) {
        fluid_particle_positions.push_back(particles_.positions[i]);
    }
    return fluid_particle_positions;
}

void SimulateSystem::simulate() {
    calculate_predicted_positions(time_step_);

    // find neighbourhood
    std::unordered_map<glm::ivec3, std::vector<int>> neighbors;
    for (int i = 0; i < particles_.size(); i++) {
        glm::ivec3 cell_index = particles_.predicted_positions[i] / (float)sph_radius_;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    neighbors[cell_index + glm::ivec3(x, y, z)].push_back(i);
                }
            }
        }
    }

    // solve density constraint
    unsigned int iter = 0;
    while (iter < 3) {
        auto lambdas = calculate_lagrange_multiplier(neighbors);
        solve_constraint(lambdas, neighbors);
        ++iter;
    }

    update_particles(time_step_);
}

void SimulateSystem::calculate_predicted_positions(float delta_time) {
    const glm::vec3 gravity(0.0f, -9.81f, 0.0f);

    for (int i = 0; i < fluid_particles_number_; i++) {
        particles_.velocities[i] += gravity * delta_time;
        particles_.predicted_positions[i] = particles_.positions[i] + particles_.velocities[i] * delta_time;

        auto &position = particles_.predicted_positions[i];
        if (position.x < container_start_.x - particle_radius_) {
            position.x = container_start_.x - particle_radius_;
        }
        if (position.x > container_end_.x + particle_radius_) {
            position.x = container_end_.x + particle_radius_;
        }

        if (position.y < container_start_.y - particle_radius_) {
            position.y = container_start_.y - particle_radius_;
        }
        if (position.y > container_end_.y + particle_radius_) {
            position.y = container_end_.y + particle_radius_;
        }

        if (position.z < container_start_.z - particle_radius_) {
            position.z = container_start_.z - particle_radius_;
        }
        if (position.z > container_end_.z + particle_radius_) {
            position.z = container_end_.z + particle_radius_;
        }
    }
}

std::vector<float> SimulateSystem::calculate_lagrange_multiplier(std::unordered_map<glm::ivec3, std::vector<int>> &neighbors) {
    std::vector<float> multipliers;
    for (int i = 0; i < fluid_particles_number_; i++) {
        glm::ivec3 cell_index = particles_.predicted_positions[i] / sph_radius_;

        // calculate density
        float density = 0;
        for (const auto &neighbor : neighbors[cell_index]) {
            density += inv_density_ * SplineInterpolation::poly6_kernel(
                                          particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
        }

        // calculate lagrange multiplier
        const float eps = 1000;
        const float constraint = std::max(density - 1.0, 0.0);
        float lambda = 0.0;

        // if (constraint != 0.0) {
            float sum_grad_cj = 0.0;
            glm::vec3 grad_ci(0.0);

            for (const auto &neighbor : neighbors[cell_index]) {
                glm::vec3 grad_cj = inv_density_ * SplineInterpolation::poly6_kernal_grade(
                                                       particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
                grad_ci += grad_cj;
                if (i != neighbor)
                    sum_grad_cj += pow(glm::length(grad_cj), 2.0);
            }
            sum_grad_cj += pow(glm::length(grad_ci), 2.0);
            lambda = -constraint / (sum_grad_cj + eps);
        // }
        multipliers.push_back(lambda);
    }
    return multipliers;
}

void SimulateSystem::solve_constraint(std::vector<float> &lambdas, std::unordered_map<glm::ivec3, std::vector<int>> &neighbors) {
    for (int i = 0; i < fluid_particles_number_; i++) {
        glm::ivec3 cell_index = particles_.predicted_positions[i] / sph_radius_;
        for (const auto &neighbor : neighbors[cell_index]) {
            glm::vec3 grad_cj = inv_density_ * SplineInterpolation::poly6_kernal_grade(
                                                   particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
            particles_.predicted_positions[i] += (lambdas[i] + lambdas[neighbor]) * grad_cj;
        }
    }
}

void SimulateSystem::update_particles(float delta_time) {
    for (int i = 0; i < particles_.size(); i++) {
        particles_.velocities[i] = (1.0f / delta_time) * (particles_.predicted_positions[i] - particles_.positions[i]);
        particles_.positions[i] = particles_.predicted_positions[i];
    }
}