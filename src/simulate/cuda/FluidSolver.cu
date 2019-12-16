#include "FluidSolver.cuh"

#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

using namespace simulate::cuda;

void simulate::cuda::print_gpu_info() {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    std::cout << "gpu name: " << dev_prop.name << std::endl;
    std::cout << "multiProcessorCount:" << dev_prop.multiProcessorCount << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << dev_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << dev_prop.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "maxThreadsPerBlock:" << dev_prop.maxThreadsPerBlock << std::endl;
    std::cout << "sharedMemPerBlock:" << dev_prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
}

void simulate::cuda::check_cuda_error(const char *error_msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error : "
                  << error_msg << " : "
                  << "(" << static_cast<int>(err) << ") "
                  << cudaGetErrorString(err) << ".\n";
    }
}

void FluidSolverCuda::set_parameters(Parameters &params) {
    cudaMemcpyToSymbol(k_params, &params, sizeof(Parameters));
    cudaDeviceSynchronize();
}

void FluidSolverCuda::malloc() {
    cudaMalloc((void **)&positions, particle_number * sizeof(float4));
    cudaMalloc((void **)&predicted_positions, particle_number * sizeof(float4));
    cudaMalloc((void **)&velocities, particle_number * sizeof(float4));

    cudaMalloc((void **)&cell_ids, particle_number * sizeof(unsigned int));
    cudaMalloc((void **)&cell_start, cell_number * sizeof(unsigned int));
    cudaMalloc((void **)&cell_end, cell_number * sizeof(unsigned int));
}

void FluidSolverCuda::simulate(float delta_time) {
    unsigned int threads_number = 1024;
    unsigned int blocks_number = (particle_number % threads_number != 0) ? (particle_number / threads_number + 1) : (particle_number / threads_number);

    calculate_predicted_positions<<<blocks_number, threads_number>>>(
        positions, velocities, predicted_positions, delta_time, particle_number);

    // find neighbourhood
    {
        calculate_cell_ids<<<blocks_number, threads_number>>>(
            cell_ids, predicted_positions, particle_number);

        sort_particles_by_cell_ids(
            positions, velocities, predicted_positions, cell_ids, particle_number);

        cudaMemset(cell_start, 0xffffffff, cell_number * sizeof(unsigned int));
        unsigned int memSize = sizeof(unsigned int) * (threads_number + 1);
        calculate_cell_range<<<blocks_number, threads_number, memSize>>>(
            cell_ids, cell_start, cell_end, particle_number);
    }

    // solve density constraint
    unsigned int iter = 0;
    while (iter < 3) {
        calculate_lagrange_multiplier<<<blocks_number, threads_number>>>(
            velocities, predicted_positions,
            cell_start, cell_end,
            particle_number);

        solve_constraint<<<blocks_number, threads_number>>>(
            velocities, predicted_positions,
            cell_start, cell_end,
            particle_number);
        ++iter;
    }

    update_particles<<<blocks_number, threads_number>>>(
        positions, velocities, predicted_positions, 1.0f / delta_time, particle_number);
}