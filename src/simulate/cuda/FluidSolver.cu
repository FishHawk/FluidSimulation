#include "FluidSolver.cuh"
#include "Kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>

using namespace simulate::cuda;

void print_gpu_info() {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    std::cout << "gpu name: " << dev_prop.name << std::endl;
    std::cout << "multiProcessorCount:" << dev_prop.multiProcessorCount << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << dev_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << dev_prop.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "maxThreadsPerBlock:" << dev_prop.maxThreadsPerBlock << std::endl;
    std::cout << "sharedMemPerBlock:" << dev_prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
}

void check_cuda_error(const char *error_msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error : "
                  << error_msg << " : "
                  << "(" << static_cast<int>(err) << ") "
                  << cudaGetErrorString(err) << ".\n";
    }
}

void setParameters(SimulateParams *hostParams) {
    cudaMemcpyToSymbol(params, hostParams, sizeof(SimulateParams));
}

void FluidSolverCuda::malloc() {
    cudaMalloc((void **)&positions, particles_number * sizeof(float4));
    cudaMalloc((void **)&predicted_positions, particles_number * sizeof(float4));
    cudaMalloc((void **)&velocities, particles_number * sizeof(float4));
    cudaMalloc((void **)&delta_positions, particles_number * sizeof(float3));

    cudaMalloc((void **)&cell_ids, particles_number * sizeof(unsigned int));
    auto gridSize = make_uint3(20, 40, 20);
    cudaMalloc((void **)&cell_start, gridSize.x * gridSize.y * gridSize.z * sizeof(unsigned int));
    cudaMalloc((void **)&cell_end, gridSize.x * gridSize.y * gridSize.z * sizeof(unsigned int));
}

void FluidSolverCuda::simulate(float delta_time) {
    unsigned int threads_number = 512;
    unsigned int blocks_number = (particles_number % threads_number != 0) ? (particles_number / threads_number + 1) : (particles_number / threads_number);

    calculate_predicted_positions<<<blocks_number, threads_number>>>(
        positions, velocities, predicted_positions, delta_time, particles_number);

    // find neighbourhood.
    {
        // calculate grid Hash.
        calculate_cell_ids<<<blocks_number, threads_number>>>(
            cell_ids, predicted_positions, particles_number);
        cudaDeviceSynchronize();

        // sort particles based on hash value.
        sort_particles_by_cell_ids(
            positions, velocities, predicted_positions, cell_ids, particles_number);
        cudaDeviceSynchronize();

        // find start index and end index of each cell.
        cudaMemset(cell_start, 0xffffffff, params.m_numGridCells * sizeof(unsigned int));
        unsigned int memSize = sizeof(unsigned int) * (threads_number + 1);
        calculate_cell_range<<<blocks_number, threads_number, memSize>>>(
            cell_ids, cell_start, cell_end, particles_number);
    }

    // density constraint.
    unsigned int iter = 0;
    while (iter < 3) {
        // calculate density and lagrange multiplier.
        calculate_lagrange_multiplier<<<blocks_number, threads_number>>>(
            velocities, predicted_positions,
            cell_ids, cell_start, cell_end,
            particles_number, params.m_numGridCells);

        // calculate delta position.
        calculate_delta_positions<<<blocks_number, threads_number>>>(
            velocities, predicted_positions, delta_positions,
            cell_ids, cell_start, cell_end,
            particles_number);

        // add delta position.
        correct_predicted_positions<<<blocks_number, threads_number>>>(
            predicted_positions, delta_positions, particles_number);
        ++iter;
    }

    update_particles<<<blocks_number, threads_number>>>(
        positions, velocities, predicted_positions, 1.0f / delta_time, particles_number);
}