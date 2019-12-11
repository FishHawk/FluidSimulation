#ifndef SIMULATE_CUDA_KERNEL_CUH
#define SIMULATE_CUDA_KERNEL_CUH

#include <math.h>

#include <cooperative_groups.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "FluidSolver.hpp"

using namespace cooperative_groups;
using namespace simulate::cuda;

__constant__ FluidSolverCuda::Parameters k_params;

// util functions
inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float wPoly6(const float3 &r) {
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > k_params.sph_radius_2 || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = k_params.sph_radius_2 - lengthSquared;
    return k_params.poly6_coff * iterm * iterm * iterm;
}

__device__ float3 wSpikyGrad(const float3 &r) {
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3 ret = {0.0f, 0.0f, 0.0f};
    if (lengthSquared > k_params.sph_radius_2 || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float iterm = k_params.sph_radius - length;
    float coff = k_params.spiky_grad_coff * iterm * iterm / length;
    ret.x = coff * r.x;
    ret.y = coff * r.y;
    ret.z = coff * r.z;
    return ret;
}

__device__ int3 calculate_cell_position(float3 p) {
    int3 gridPos;
    gridPos.x = floor((p.x - k_params.container_start.x) / k_params.sph_radius);
    gridPos.y = floor((p.y - k_params.container_start.y) / k_params.sph_radius);
    gridPos.z = floor((p.z - k_params.container_start.z) / k_params.sph_radius);
    return gridPos;
}

__device__ unsigned int calculate_cell_id(int3 gridPos) {
    gridPos.x = gridPos.x & (k_params.grid_size.x - 1);
    gridPos.y = gridPos.y & (k_params.grid_size.y - 1);
    gridPos.z = gridPos.z & (k_params.grid_size.z - 1);
    return gridPos.z * k_params.grid_size.x * k_params.grid_size.y + gridPos.y * k_params.grid_size.x + gridPos.x;
}

// kernel functions
__global__ void calculate_predicted_positions(
    float4 *positions,
    float4 *velocities,
    float4 *predicted_positions,
    float delta_time,
    unsigned int particles_number) {
    // bound check
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    float4 velocity = velocities[index];
    float4 position = positions[index];
    float3 n_velocity;
    float3 n_position;

    // predict position
    n_velocity.x = velocity.x;
    n_velocity.y = velocity.y + delta_time * -9.8f;
    n_velocity.z = velocity.z;
    n_position.x = position.x + delta_time * n_velocity.x;
    n_position.y = position.y + delta_time * n_velocity.y;
    n_position.z = position.z + delta_time * n_velocity.z;

    // collision with walls.
    if (n_position.x < k_params.container_start.x + k_params.particle_radius) {
        n_position.x = k_params.container_start.x + k_params.particle_radius;
    }
    if (n_position.x > k_params.container_end.x - k_params.particle_radius) {
        n_position.x = k_params.container_end.x - k_params.particle_radius;
    }

    if (n_position.y < k_params.container_start.y + k_params.particle_radius) {
        n_position.y = k_params.container_start.y + k_params.particle_radius;
    }
    if (n_position.y > k_params.container_end.y - k_params.particle_radius) {
        n_position.y = k_params.container_end.y - k_params.particle_radius;
    }

    if (n_position.z < k_params.container_start.z + k_params.particle_radius) {
        n_position.z = k_params.container_start.z + k_params.particle_radius;
    }
    if (n_position.z > k_params.container_end.z - k_params.particle_radius) {
        n_position.z = k_params.container_end.z - k_params.particle_radius;
    }

    predicted_positions[index] = {n_position.x, n_position.y, n_position.z, position.w};
}

__global__ void calculate_cell_ids(
    unsigned int *cell_ids,
    float4 *positions,
    unsigned int particles_number) {
    // bound check
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    float4 position = positions[index];
    int3 cell_position = calculate_cell_position(make_float3(position.x, position.y, position.z));
    cell_ids[index] = calculate_cell_id(cell_position);
}

void sort_particles_by_cell_ids(
    float4 *positions,
    float4 *velocities,
    float4 *predicted_positions,
    unsigned int *cell_ids,
    unsigned int particles_number) {

    thrust::device_ptr<float4> ptrPos(positions);
    thrust::device_ptr<float4> ptrVel(velocities);
    thrust::device_ptr<float4> ptrPredictedPos(predicted_positions);
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(cell_ids),
        thrust::device_ptr<unsigned int>(cell_ids + particles_number),
        thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptrPredictedPos)));
}

__global__ void calculate_cell_range(
    unsigned int *cell_ids,
    unsigned int *cell_start,
    unsigned int *cell_end,
    unsigned int particles_number) {
    thread_block cta = this_thread_block();
    extern __shared__ unsigned int sharedHash[];
    unsigned int hashValue;

    // bound check
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    hashValue = cell_ids[index];
    sharedHash[threadIdx.x + 1] = hashValue;
    if (index > 0 && threadIdx.x == 0)
        sharedHash[0] = cell_ids[index - 1];

    sync(cta);

    if (index == 0 || hashValue != sharedHash[threadIdx.x]) {
        cell_start[hashValue] = index;
        if (index > 0)
            cell_end[sharedHash[threadIdx.x]] = index;
    }
    if (index == particles_number - 1)
        cell_end[hashValue] = index + 1;
}

__global__ void calculate_lagrange_multiplier(
    float4 *velocities,
    float4 *predicted_positions,
    unsigned int *cell_start,
    unsigned int *cell_end,
    unsigned int particles_number) {
    // bound check
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    float3 position_i{predicted_positions[index].x,
                      predicted_positions[index].y,
                      predicted_positions[index].z};
    int3 cell_position = calculate_cell_position(position_i);

    float density = 0.0f;
    float gradSquaredSum_j = 0.0f;
    float gradSquaredSumTotal = 0.0f;
    float3 gradSum_i = {0.0f, 0.0f, 0.0f};
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {

                int3 cell_position_neighbor{cell_position.x + x, cell_position.y + y, cell_position.z + z};
                unsigned int cell_id_neighbor = calculate_cell_id(cell_position_neighbor);
                unsigned int start_index = cell_start[cell_id_neighbor];
                if (start_index == 0xffffffff)
                    continue;
                unsigned int end_index = cell_end[cell_id_neighbor];

                for (unsigned int j = start_index; j < end_index; ++j) {
                    float4 position_j = predicted_positions[j];
                    float3 r = {position_i.x - position_j.x, position_i.y - position_j.y, position_i.z - position_j.z};
                    density += wPoly6(r);
                    float3 grad_cj = wSpikyGrad(r);
                    grad_cj.x *= k_params.inv_density;
                    grad_cj.y *= k_params.inv_density;
                    grad_cj.z *= k_params.inv_density;

                    gradSum_i.x += grad_cj.x;
                    gradSum_i.y += grad_cj.y;
                    gradSum_i.z += grad_cj.z;
                    if (j != index)
                        gradSquaredSum_j += (grad_cj.x * grad_cj.x + grad_cj.y * grad_cj.y + grad_cj.z * grad_cj.z);
                }
            }
        }
    }
    gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;

    // density constraint.
    float constraint = density * k_params.inv_density - 1.0f;
    float lambda = -constraint / (gradSquaredSumTotal + 1000.0f);
    velocities[index].w = lambda;
    predicted_positions[index].w = density;
}

__global__ void solve_constraint(
    float4 *velocities,
    float4 *predicted_positions,
    unsigned int *cell_start,
    unsigned int *cell_end,
    unsigned int particles_number) {
    // bound check
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    float3 position_i{predicted_positions[index].x,
                      predicted_positions[index].y,
                      predicted_positions[index].z};
    float lambda_i = velocities[index].w;
    int3 cell_position = calculate_cell_position(position_i);

    float3 delta_position{0.0f, 0.0f, 0.0f};
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {

                int3 cell_position_neighbor{cell_position.x + x, cell_position.y + y, cell_position.z + z};
                unsigned int cell_id_neighbor = calculate_cell_id(cell_position_neighbor);
                unsigned int start_index = cell_start[cell_id_neighbor];
                if (start_index == 0xffffffff)
                    continue;
                unsigned int end_index = cell_end[cell_id_neighbor];

                for (unsigned int j = start_index; j < end_index; ++j) {
                    float4 position_j = predicted_positions[j];
                    float lambda_j = velocities[j].w;
                    float3 r = {position_i.x - position_j.x, position_i.y - position_j.y, position_i.z - position_j.z};
                    // float corrTerm = wPoly6(r) * params.m_oneDivWPoly6;
                    // float coff = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
                    float coff = lambda_i + lambda_j;
                    float3 grad = wSpikyGrad(r);
                    delta_position.x += coff * grad.x;
                    delta_position.y += coff * grad.y;
                    delta_position.z += coff * grad.z;
                }
            }
        }
    }

    predicted_positions[index].x += delta_position.x * k_params.inv_density;
    predicted_positions[index].y += delta_position.y * k_params.inv_density;
    predicted_positions[index].z += delta_position.z * k_params.inv_density;
}

__global__ void update_particles(
    float4 *positions,
    float4 *velocities,
    float4 *predicted_positions,
    float inv_delta_time,
    unsigned int particles_number) {
    // bound check
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_number)
        return;

    float4 position = positions[index];
    float4 n_position = predicted_positions[index];

    velocities[index].x = (n_position.x - position.x) * inv_delta_time;
    velocities[index].y = (n_position.y - position.y) * inv_delta_time;
    velocities[index].z = (n_position.z - position.z) * inv_delta_time;

    positions[index].x = n_position.x;
    positions[index].y = n_position.y;
    positions[index].z = n_position.z;
}

#endif