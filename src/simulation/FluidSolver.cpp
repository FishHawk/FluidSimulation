#include "FluidSolver.hpp"

#include "Kernel.hpp"

namespace std {
template <>
struct less<glm::ivec3> {
    bool operator()(const glm::ivec3 &lhs, const glm::ivec3 &rhs) const {
        return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
    }
};
}  // namespace std

FluidSolver::FluidSolver() {
}

void FluidSolver::update_time_step() {
    static const float max_time_step = 0.005, min_time_step = 0.0001;

    float max_mag = 0.1;
    for (int i = 0; i < particles_.size(); ++i) {
        const auto &velocity = particles_.velocities[i];
        const auto &acceleration = particles_.accelerations[i];
        const float mag = glm::length(velocity + time_step_ * acceleration);

        if (mag > max_mag)
            max_mag = mag;
    }

    const float cfl_factor = 0.4;
    const float diameter = 2.0 * particle_radius_;
    time_step_ = cfl_factor * (diameter / max_mag);

    time_step_ = std::min(time_step_, max_time_step);
    // time_step_ = std::max(time_step_, min_time_step);
}

void FluidSolver::setup_model(const std::vector<glm::vec3> &fluid_particles,
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
    std::map<glm::ivec3, std::vector<int>> neighbors;
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
            delta += Kernel::poly6_kernel(boundary_particles[i] - boundary_particles[neighbor], sph_radius_);
        }
        particles_.add(fluid_density_ / delta, boundary_particles[i]);
    }

    fluid_particals_number_ = fluid_particles.size();
    boundary_particals_number_ = boundary_particles.size();
}

void FluidSolver::simulation() {
    auto time_now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<float>(time_now - time_point_).count();
    if (duration < time_step_) {
        return;
    } else {
        // update time
        time_point_ = time_now;

        reset_acceleration();
        update_time_step();

        update_particles_ignore_constraint(time_step_);

        constraint_projection();

        correct_velocity(time_step_);

        // // compute viscoity.
        // computeXSPHViscosity();
        // computeVorticityConfinement();
    }
}

std::vector<glm::vec3> FluidSolver::get_partical_position() {
    std::vector<glm::vec3> fluid_particle_positions;
    // for (int i = 0; i < particles_.size(); i++) {
    for (int i = 0; i < fluid_particals_number_; i++) {
        fluid_particle_positions.push_back(particles_.positions[i]);
    }
    return fluid_particle_positions;
}

void FluidSolver::reset_acceleration() {
    const glm::vec3 gravity(0.0f, -9.81f, 0.0f);
    // const glm::vec3 gravity(0.0f, -0.01f, 0.0f);

    for (int i = 0; i < fluid_particals_number_; i++) {
        particles_.accelerations[i] = gravity;
    }
}

void FluidSolver::update_particles_ignore_constraint(float delta_time) {
    for (int i = 0; i < fluid_particals_number_; i++) {
        if (particles_.masses[i] != 0) {
            particles_.last_positions[i] = particles_.old_positions[i];
            particles_.old_positions[i] = particles_.positions[i];
            particles_.velocities[i] += particles_.accelerations[i] * delta_time;
            particles_.positions[i] += particles_.velocities[i] * delta_time;
        }
    }
}

void FluidSolver::constraint_projection() {
    std::map<glm::ivec3, std::vector<int>> neighbors;
    for (int i = 0; i < particles_.size(); i++) {
        glm::ivec3 cell_index = particles_.positions[i] / (float)sph_radius_;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    neighbors[cell_index + glm::ivec3(x, y, z)].push_back(i);
                }
            }
        }
    }

    int iter = 0;
    while (iter < 5) {
        // calculate density and lagrange multiplier.
        auto densities = calculate_fluid_density(neighbors);
        auto lambdas = calculate_lagrange_multiplier(densities, neighbors);
        // std::cout<<densities[0]<<"  ";
        // std::cout<<densities[14*14*7]<<"  ";
        // std::cout<<densities[2]<<"  ";
        // std::cout<<std::endl;

        // perform density constraint.
        solve_constraint(lambdas, neighbors);

        ++iter;
    }
}

void FluidSolver::correct_velocity(float delta_time) {
    for (int i = 0; i < particles_.size(); i++) {
        if (particles_.masses[i] != 0)
            particles_.velocities[i] = (1.0f / delta_time) *
                                       (particles_.positions[i] - particles_.old_positions[i]);
    }
}

std::vector<double> FluidSolver::calculate_fluid_density(std::map<glm::ivec3, std::vector<int>> &neighbors) {
    std::vector<double> densities;
    for (int i = 0; i < fluid_particals_number_; i++) {
        glm::ivec3 cell_index = particles_.positions[i] / (float)sph_radius_;
        double density = 0;
        for (const auto &neighbor : neighbors[cell_index]) {
            density += particles_.masses[neighbor] *
                       Kernel::poly6_kernel(particles_.positions[i] - particles_.positions[neighbor], sph_radius_);
        }
        densities.push_back(density);
    }
    return densities;
}

std::vector<double> FluidSolver::calculate_lagrange_multiplier(std::vector<double> &densities, std::map<glm::ivec3, std::vector<int>> &neighbors) {
    std::vector<double> multipliers;
    for (int i = 0; i < fluid_particals_number_; i++) {
        const double eps = 1.0e-6;
        const double constraint = std::max(densities[i] / fluid_density_ - 1.0, 0.0);
        double lambda = 0.0;

        if (constraint != 0.0) {
            double sum_grad_cj = 0.0;
            glm::vec3 grad_ci(0.0);

            glm::ivec3 cell_index = particles_.positions[i] / (float)sph_radius_;
            for (const auto &neighbor : neighbors[cell_index]) {
                glm::vec3 grad_cj = static_cast<float>(particles_.masses[neighbor] / fluid_density_) *
                                    Kernel::poly6_kernal_grade(particles_.positions[i] - particles_.positions[neighbor], sph_radius_);
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

void FluidSolver::solve_constraint(std::vector<double> &lambdas, std::map<glm::ivec3, std::vector<int>> &neighbors) {
    for (int i = 0; i < fluid_particals_number_; i++) {
        auto delta_pos = glm::vec3(0.0f);

        glm::ivec3 cell_index = particles_.positions[i] / (float)sph_radius_;
        for (const auto &neighbor : neighbors[cell_index]) {
            glm::vec3 grad_cj = static_cast<float>(particles_.masses[neighbor] / fluid_density_) *
                                Kernel::poly6_kernal_grade(particles_.positions[i] - particles_.positions[neighbor], sph_radius_);
            if (neighbor < fluid_particals_number_) {
                delta_pos += static_cast<float>(lambdas[i] + lambdas[neighbor]) * grad_cj;
            } else {
                delta_pos += static_cast<float>(lambdas[i]) * grad_cj;
            }
        }

        particles_.positions[i] += delta_pos;
    }
}