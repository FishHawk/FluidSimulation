#include "SimulateSystem.hpp"

#include <cuda_runtime.h>

using namespace simulate::cuda;

void SimulateSystem::setup_model(const std::vector<glm::vec3> &fluid_particles,
                                 const std::vector<glm::vec3> &boundary_particles) {
    fluid_particles_number_ = fluid_particles.size();
    boundary_particles_number_ = boundary_particles.size();

    for (const auto &pos : fluid_particles) {
        positions_.push_back(glm::vec4(pos, 0));
    }
    // for (const auto &pos : boundary_particles) {
    //     positions_.push_back(glm::vec4(pos, 0));
    // }

    solver_.particles_number = fluid_particles_number_;
    solver_.malloc();
    cudaMemcpy(solver_.positions, positions_.data(), fluid_particles_number_ * sizeof(float4), cudaMemcpyHostToDevice);

    auto gridSize = make_uint3(20, 40, 20);

    // particles and grid.
    m_params.m_gridSize = gridSize;
    m_params.m_numGridCells = gridSize.x * gridSize.y * gridSize.z;

    // smooth kernel radius.
    m_params.m_particleRadius = 0.025;
    m_params.m_sphRadius = 4.0 * 0.025;
    m_params.m_sphRadiusSquared = m_params.m_sphRadius * m_params.m_sphRadius;

    // lagrange multiplier eps.
    m_params.m_lambdaEps = 1000.0f;

    // fluid reset density.
    m_params.m_restDensity = 1.0f / (8.0f * powf(0.025, 3.0f));
    m_params.m_invRestDensity = 1.0f / m_params.m_restDensity;

    // sph kernel function coff.
    m_params.m_poly6Coff = 315.0f / (64.0f * M_PI * powf(m_params.m_sphRadius, 9.0));
    m_params.m_spikyGradCoff = -45.0f / (M_PI * powf(m_params.m_sphRadius, 6.0));

    // grid cells.
    float cellSize = m_params.m_sphRadius;
    m_params.m_cellSize = make_float3(cellSize, cellSize, cellSize);
    m_params.m_worldOrigin = {-1.0f, 0.0f, -1.0f};

    m_params.m_oneDivWPoly6 = 1.0f / (m_params.m_poly6Coff *
                                      pow(m_params.m_sphRadiusSquared - pow(0.1 * m_params.m_sphRadius, 2.0), 3.0));
}

void SimulateSystem::simulate() {
    auto time_now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<float>(time_now - time_point_).count();
    if (duration < time_step_) {
        return;
    } else {
        std::cout << duration / time_step_ << std::endl;
        // update time
        time_point_ = time_now;
        // update constants
        setParameters(&m_params);

        solver_.simulate(time_step_);
    }
}

std::vector<glm::vec3> SimulateSystem::get_partical_position() {
    cudaMemcpy(positions_.data(), solver_.positions, positions_.size() * sizeof(float4), cudaMemcpyDeviceToHost);
    std::vector<glm::vec3> fluid_postitions;
    for (int i = 0; i < fluid_particles_number_; i++) {
        const auto &position = positions_[i];
        fluid_postitions.push_back(glm::vec3(position.x, position.y, position.z));
    }
    return fluid_postitions;
}
