#include "FluidSolver.hpp"

FluidSolver::FluidSolver() {
}

void FluidSolver::update_time_step() {
    static const float max_time_step = 0.005, min_time_step = 0.0001;

    float max_mag = 0.1;
    for (int i = 0; i < particles_.size(); ++i) {
        const auto &velocity = particles_.velocities[i];
        const auto &acceleration = particles_.accelerations[i];
        const float mag = pow(glm::length(velocity + time_step_ * acceleration), 2.0);

        if (mag > max_mag)
            max_mag = mag;
    }

    const float cfl_factor = 1.0;
    const float diameter = 2.0 * particle_radius_;
    time_step_ = cfl_factor * 0.40 * (diameter / sqrt(max_mag));

    time_step_ = std::min(time_step_, max_time_step);
    time_step_ = std::max(time_step_, min_time_step);
}

void FluidSolver::setup_model(const std::vector<glm::vec3> &fluid_particles,
                              const std::vector<glm::vec3> &boundary_particles) {
    time_point_ = std::chrono::system_clock::now();

    particles_.clear();
    particles_.reserve(fluid_particles.size() + boundary_particles.size());

    double diameter = 2.0 * particle_radius_;
    double volume = diameter * diameter * diameter * 0.8;
    double mass = volume * fluid_density_;
    for (const auto &position : fluid_particles) {
        particles_.add(mass, position);
    }

    for (const auto &position : boundary_particles) {
        particles_.add(0, position);
    }
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

        // // search each particle's neighborhood.
        // m_neighborSearch->neighborhoodSearch(m_particles.getPositionGroup(),
        //                                      0, nFluidParticles);

        // // perform density constraint.
        // constraintProjection();

        correct_velocity(time_step_);

        // // compute viscoity.
        // computeXSPHViscosity();
        // computeVorticityConfinement();
    }
}

std::vector<glm::vec3> FluidSolver::get_partical_position() {
    std::vector<glm::vec3> fluid_particle_positions;
    for (int i = 0; i < particles_.size(); i++) {
        // if (particles_.masses[i] == 0) {
            fluid_particle_positions.push_back(particles_.positions[i]);
        // }
    }
    return fluid_particle_positions;
}

void FluidSolver::reset_acceleration() {
    const glm::vec3 gravity(0.0f, -9.81f, 0.0f);
    // const glm::vec3 gravity(0.0f, -0.01f, 0.0f);

    for (int i = 0; i < particles_.size(); i++) {
        if (particles_.masses[i] != 0) {
            particles_.accelerations[i] = gravity;
        }
    }
}

void FluidSolver::update_particles_ignore_constraint(float delta_time) {
    for (int i = 0; i < particles_.size(); i++) {
        if (particles_.masses[i] != 0) {
            particles_.last_positions[i] = particles_.old_positions[i];
            particles_.old_positions[i] = particles_.positions[i];
            particles_.velocities[i] += particles_.accelerations[i] * delta_time;
            particles_.positions[i] += particles_.velocities[i] * delta_time;
        }
    }
}

void FluidSolver::correct_velocity(float delta_time) {
    for (int i = 0; i < particles_.size(); i++) {
        if (particles_.masses[i] != 0)
            particles_.velocities[i] = (1.0f / delta_time) *
                                       (particles_.positions[i] - particles_.old_positions[i]);
    }
}