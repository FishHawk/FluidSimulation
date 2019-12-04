#include "SimulateSystem.hpp"

#include "SplineInterpolation.hpp"
#include "glm/gtx/hash.hpp"

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

void SimulateSystem::setup_model(const std::vector<glm::vec3> &fluid_particles,
                                 const std::vector<glm::vec3> &boundary_particles) {
    time_point_ = std::chrono::system_clock::now();

    particles_.clear();
    particles_.reserve(fluid_particles.size() + boundary_particles.size());

    // init fluid particles
    double diameter = 2.0 * particle_radius_;
    double volume = diameter * diameter * diameter * 0.8;
    double mass = volume * fluid_density_;
    for (const auto &position : fluid_particles) {
        particles_.add(mass, position);
    }

    // init boundary particles
    std::unordered_map<glm::ivec3, std::vector<int>> neighbors;
    for (int i = 0; i < boundary_particles.size(); i++) {
        glm::ivec3 cell_index = boundary_particles[i] / (float)sph_radius_;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    neighbors[cell_index + glm::ivec3(x, y, z)].push_back(i);
                }
            }
        }
    }

    for (int i = 0; i < boundary_particles.size(); i++) {
        double delta = 0;
        glm::ivec3 cell_index = boundary_particles[i] / (float)sph_radius_;
        for (const auto &neighbor : neighbors[cell_index]) {
            delta += SplineInterpolation::poly6_kernel(boundary_particles[i] - boundary_particles[neighbor], sph_radius_);
        }
        particles_.add(fluid_density_ / delta, boundary_particles[i]);
    }

    fluid_particles_number_ = fluid_particles.size();
    boundary_particles_number_ = boundary_particles.size();
}

std::vector<glm::vec3> SimulateSystem::get_particle_position() {
    std::vector<glm::vec3> fluid_particle_positions;
    // for (int i = 0; i < particles_.size(); i++) {
    for (int i = 0; i < fluid_particles_number_; i++) {
        fluid_particle_positions.push_back(particles_.positions[i]);
    }
    return fluid_particle_positions;
}

void SimulateSystem::simulate() {
    auto time_now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<float>(time_now - time_point_).count();
    if (duration < time_step_) {
        return;
    } else {
        // update time
        time_point_ = time_now;

        calculate_predicted_positions(time_step_);
        update_time_step();

        // find neighbors
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

        // density constraint
        unsigned int iter = 0;
        while (iter < 3) {
            auto lambdas = calculate_lagrange_multiplier(neighbors);
            solve_constraint(lambdas, neighbors);
            ++iter;
        }

        update_particles(time_step_);
    }
}

void SimulateSystem::calculate_predicted_positions(float delta_time) {
    const glm::vec3 gravity(0.0f, -9.81f, 0.0f);

    for (int i = 0; i < fluid_particles_number_; i++) {
        if (particles_.masses[i] != 0) {
            particles_.velocities[i] += gravity * delta_time;
            particles_.predicted_positions[i] = particles_.positions[i] + particles_.velocities[i] * delta_time;
        }
    }
}

void SimulateSystem::update_time_step() {
    static const float max_time_step = 0.005, min_time_step = 0.0001;

    float max_v = 0.1;
    for (int i = 0; i < particles_.size(); ++i) {
        const float v = glm::length(particles_.velocities[i]);
        if (v > max_v)
            max_v = v;
    }

    const float cfl_factor = 0.4;
    const float diameter = 2.0 * particle_radius_;
    time_step_ = cfl_factor * (diameter / max_v);

    time_step_ = std::min(time_step_, max_time_step);
    // time_step_ = std::max(time_step_, min_time_step);
}

std::vector<double> SimulateSystem::calculate_lagrange_multiplier(std::unordered_map<glm::ivec3, std::vector<int>> &neighbors) {
    std::vector<double> multipliers;
    for (int i = 0; i < fluid_particles_number_; i++) {
        glm::ivec3 cell_index = particles_.predicted_positions[i] / (float)sph_radius_;

        // calculate density
        double density = 0;
        for (const auto &neighbor : neighbors[cell_index]) {
            density += particles_.masses[neighbor] *
                       SplineInterpolation::poly6_kernel(particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
        }

        // calculate lagrange multiplier
        const double eps = 1.0e-6;
        const double constraint = std::max(density / fluid_density_ - 1.0, 0.0);
        double lambda = 0.0;

        if (constraint != 0.0) {
            double sum_grad_cj = 0.0;
            glm::vec3 grad_ci(0.0);

            for (const auto &neighbor : neighbors[cell_index]) {
                glm::vec3 grad_cj = static_cast<float>(particles_.masses[neighbor] / fluid_density_) *
                                    SplineInterpolation::poly6_kernal_grade(particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
                sum_grad_cj += pow(glm::length(grad_cj), 2.0);
                grad_ci += grad_cj;
            }
            sum_grad_cj += pow(glm::length(grad_ci), 2.0);
            lambda = -constraint / (sum_grad_cj + eps);
        }
        multipliers.push_back(lambda);
    }
    return multipliers;
}

void SimulateSystem::solve_constraint(std::vector<double> &lambdas, std::unordered_map<glm::ivec3, std::vector<int>> &neighbors) {
    for (int i = 0; i < fluid_particles_number_; i++) {
        auto delta_pos = glm::vec3(0.0f);

        glm::ivec3 cell_index = particles_.predicted_positions[i] / (float)sph_radius_;
        for (const auto &neighbor : neighbors[cell_index]) {
            glm::vec3 grad_cj = static_cast<float>(particles_.masses[neighbor] / fluid_density_) *
                                SplineInterpolation::poly6_kernal_grade(particles_.predicted_positions[i] - particles_.predicted_positions[neighbor], sph_radius_);
            if (neighbor < fluid_particles_number_) {
                delta_pos += static_cast<float>(lambdas[i] + lambdas[neighbor]) * grad_cj;
            } else {
                delta_pos += static_cast<float>(lambdas[i]) * grad_cj;
            }
        }

        particles_.predicted_positions[i] += delta_pos;
    }
}

void SimulateSystem::update_particles(float delta_time) {
    for (int i = 0; i < particles_.size(); i++) {
        if (particles_.masses[i] != 0)
            particles_.velocities[i] = (1.0f / delta_time) * (particles_.predicted_positions[i] - particles_.positions[i]);
        particles_.positions[i] = particles_.predicted_positions[i];
    }
}