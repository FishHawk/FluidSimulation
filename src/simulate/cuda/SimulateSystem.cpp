#include "SimulateSystem.hpp"

#include <cuda_runtime.h>

using namespace simulate::cuda;

void SimulateSystem::set_particle_position(const std::vector<glm::vec3> &particles_initial_positions) {
    fluid_particles_number_ = particles_initial_positions.size();

    for (const auto &pos : particles_initial_positions) {
        positions_.push_back(glm::vec4(pos, 0));
        initial_positions_.push_back(glm::vec4(pos, 0));
    }
}

void SimulateSystem::reset() {
    stop();
    std::lock_guard<std::mutex> lock(m);
    cudaMemcpy(solver_.positions, initial_positions_.data(), fluid_particles_number_ * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(solver_.velocities, 0, fluid_particles_number_ * sizeof(float4));
}

void SimulateSystem::simulate() {
    solver_.simulate(time_step_);
}

void SimulateSystem::apply() {
    // update constants
    FluidSolverCuda::Parameters params;

    // container
    params.container_start = make_float3(container_start_.x, container_start_.y, container_start_.z);
    params.container_end = make_float3(container_end_.x, container_end_.y, container_end_.z);

    // radius
    params.particle_radius = particle_radius_;
    params.sph_radius = sph_radius_;
    params.sph_radius_2 = sph_radius_ * sph_radius_;

    // sph kernel function coff.
    params.poly6_coff = 315.0f / (64.0f * M_PI * powf(sph_radius_, 9.0));
    params.spiky_grad_coff = -45.0f / (M_PI * powf(sph_radius_, 6.0));

    // fluid density.
    params.density = 1.0f / (8.0f * powf(particle_radius_, 3.0f));
    params.inv_density = (8.0f * powf(particle_radius_, 3.0f));

    // grid
    auto container_size = container_end_ - container_start_;
    params.grid_size = make_uint3(
        container_size.x / sph_radius_ + 1,
        container_size.y / sph_radius_ + 1,
        container_size.z / sph_radius_ + 1);

    FluidSolverCuda::set_parameters(params);

    solver_.particle_number = fluid_particles_number_;
    solver_.cell_number = params.grid_size.x * params.grid_size.y * params.grid_size.z;
    solver_.malloc();
    cudaMemcpy(solver_.positions, positions_.data(), fluid_particles_number_ * sizeof(float4), cudaMemcpyHostToDevice);
}

std::vector<glm::vec3> SimulateSystem::get_particle_position() {
    cudaMemcpy(positions_.data(), solver_.positions, positions_.size() * sizeof(float4), cudaMemcpyDeviceToHost);
    std::vector<glm::vec3> fluid_postitions;
    for (int i = 0; i < fluid_particles_number_; i++) {
        const auto &position = positions_[i];
        fluid_postitions.push_back(glm::vec3(position.x, position.y, position.z));
    }
    return fluid_postitions;
}
