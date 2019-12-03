#ifndef SIMULATE_CUDA_KERNEL_CUH
#define SIMULATE_CUDA_KERNEL_CUH

#include <cooperative_groups.h>
#include <math.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "FluidSolver.cuh"

using namespace cooperative_groups;

__constant__ SimulateParams params;

__global__ void calculate_predicted_positions(float4 *positions, float4 *velocities, float4 *predicted_positions, float delta_time, unsigned int particles_num);
__global__ void update_particles(float4 *positions, float4 *velocities, float4 *predicted_positions, float inv_delta_time, unsigned int particles_num);

__global__ void calculate_cell_ids(unsigned int *gridParticleHash, float4 *pos, unsigned int numParticles);
__global__ void calculate_cell_range(unsigned int *gridParticleHash, unsigned int *cellStart, unsigned int *cellEnd, unsigned int numParticles);

__global__ void calculate_lagrange_multiplier(
    float4 *velocity, float4 *predictedPos,
    unsigned int *gridParticleHash, unsigned int *cellStart, unsigned int *cellEnd,
    unsigned int numParticles, unsigned int numCells);
__global__ void calculate_delta_positions(
    float4 *velocity, float4 *predictedPos, float3 *deltaPos,
    unsigned int *gridParticleHash, unsigned int *cellStart, unsigned int *cellEnd, unsigned int numParticles);
__global__ void correct_predicted_positions(
    float4 *predictedPos, float3 *deltaPos, unsigned int numParticles);

inline __host__ __device__
    float3
    cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float wPoly6(const float3 &r) {
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = params.m_sphRadiusSquared - lengthSquared;
    return params.m_poly6Coff * iterm * iterm * iterm;
}

__device__ float3 wSpikyGrad(const float3 &r) {
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3 ret = {0.0f, 0.0f, 0.0f};
    if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float iterm = params.m_sphRadius - length;
    float coff = params.m_spikyGradCoff * iterm * iterm / length;
    ret.x = coff * r.x;
    ret.y = coff * r.y;
    ret.z = coff * r.z;
    return ret;
}

__device__ int3 calcGridPosKernel(float3 p) {
    int3 gridPos;
    gridPos.x = floor((p.x - params.m_worldOrigin.x) / params.m_cellSize.x);
    gridPos.y = floor((p.y - params.m_worldOrigin.y) / params.m_cellSize.y);
    gridPos.z = floor((p.z - params.m_worldOrigin.z) / params.m_cellSize.z);
    return gridPos;
}

__device__ unsigned int calcGridHashKernel(int3 gridPos) {
    gridPos.x = gridPos.x & (params.m_gridSize.x - 1);
    gridPos.y = gridPos.y & (params.m_gridSize.y - 1);
    gridPos.z = gridPos.z & (params.m_gridSize.z - 1);
    return gridPos.z * params.m_gridSize.x * params.m_gridSize.y + gridPos.y * params.m_gridSize.x + gridPos.x;
}

__global__ void calculate_predicted_positions(float4 *positions, float4 *velocities, float4 *predicted_positions, float delta_time, unsigned int particles_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_num)
        return;

    float4 readVel = velocities[index];
    float4 readPos = positions[index];
    float3 nVel;
    float3 nPos;
    nVel.x = readVel.x;
    nVel.y = readVel.y + delta_time * -9.8f;
    nVel.z = readVel.z;
    nPos.x = readPos.x + delta_time * nVel.x;
    nPos.y = readPos.y + delta_time * nVel.y;
    nPos.z = readPos.z + delta_time * nVel.z;

    // collision with walls.
    if (nPos.x > 1.0f - params.m_particleRadius) {
        nPos.x = 1.0f - params.m_particleRadius;
        nVel.x = -0.9 * nVel.x;
    }
    if (nPos.x < -1.0f + params.m_particleRadius) {
        nPos.x = -1.0f + params.m_particleRadius;
        nVel.x = -0.9 * nVel.x;
    }

    if (nPos.y > 4.0f - params.m_particleRadius) {
        nPos.y = 4.0f - params.m_particleRadius;
        nVel.y = -0.9 * nVel.y;
    }
    if (nPos.y < -0.0f + params.m_particleRadius) {
        nPos.y = -0.0f + params.m_particleRadius;
        nVel.y = -0.9 * nVel.y;
    }

    if (nPos.z > 1.0f - params.m_particleRadius) {
        nPos.z = 1.0f - params.m_particleRadius;
        nVel.z = -0.9 * nVel.z;
    }
    if (nPos.z < -1.0f + params.m_particleRadius) {
        nPos.z = -1.0f + params.m_particleRadius;
        nVel.z = -0.9 * nVel.z;
    }

    predicted_positions[index] = {nPos.x, nPos.y, nPos.z, readPos.w};
}

__global__ void update_particles(float4 *positions, float4 *velocities, float4 *predicted_positions, float inv_delta_time, unsigned int particles_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles_num)
        return;

    float4 oldPos = positions[index];
    float4 newPos = predicted_positions[index];
    float4 readVel = velocities[index];
    float3 posDiff = {newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z};
    posDiff.x *= inv_delta_time;
    posDiff.y *= inv_delta_time;
    posDiff.z *= inv_delta_time;
    velocities[index] = {posDiff.x, posDiff.y, posDiff.z, readVel.w};
    positions[index] = {newPos.x, newPos.y, newPos.z, newPos.w};
}

__global__ void calculate_cell_ids(
    unsigned int *gridParticleHash,
    float4 *pos,
    unsigned int numParticles) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    volatile float4 curPos = pos[index];
    int3 gridPos = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z));
    unsigned int hashValue = calcGridHashKernel(gridPos);
    gridParticleHash[index] = hashValue;
}

void sort_particles_by_cell_ids(float4 *positions, float4 *velocities, float4 *predicted_positions, unsigned int *cell_ids, unsigned int particles_number) {
    thrust::device_ptr<float4> ptrPos(positions);
    thrust::device_ptr<float4> ptrVel(velocities);
    thrust::device_ptr<float4> ptrPredictedPos(predicted_positions);
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(cell_ids),
        thrust::device_ptr<unsigned int>(cell_ids + particles_number),
        thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptrPredictedPos)));
}

__global__ void calculate_cell_range(
    unsigned int *gridParticleHash, // input: sorted grid hashes
    unsigned int *cellStart,        // output: cell start index
    unsigned int *cellEnd,          // output: cell end index
    unsigned int numParticles) {
    thread_block cta = this_thread_block();
    extern __shared__ unsigned int sharedHash[];
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int hashValue;

    if (index < numParticles) {
        hashValue = gridParticleHash[index];
        sharedHash[threadIdx.x + 1] = hashValue;

        // first thread in block must load neighbor particle hash
        if (index > 0 && threadIdx.x == 0)
            sharedHash[0] = gridParticleHash[index - 1];
    }

    sync(cta);

    if (index < numParticles) {
        if (index == 0 || hashValue != sharedHash[threadIdx.x]) {
            cellStart[hashValue] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
            cellEnd[hashValue] = index + 1;
    }
}

__global__ void calculate_lagrange_multiplier(
    float4 *velocity,
    float4 *predictedPos,
    unsigned int *gridParticleHash,
    unsigned int *cellStart,
    unsigned int *cellEnd,
    unsigned int numParticles,
    unsigned int numCells) {
    // calculate current particle's density and lagrange multiplier.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float4 readPos = predictedPos[index];
    float4 readVel = velocity[index];
    float3 curPos = {readPos.x, readPos.y, readPos.z};
    int3 gridPos = calcGridPosKernel(curPos);

    float density = 0.0f;
    float gradSquaredSum_j = 0.0f;
    float gradSquaredSumTotal = 0.0f;
    float3 curGrad, gradSum_i = {0.0f, 0.0f, 0.0f};
#pragma unroll 3
    for (int z = -1; z <= 1; ++z) {
#pragma unroll 3
        for (int y = -1; y <= 1; ++y) {
#pragma unroll 3
            for (int x = -1; x <= 1; ++x) {
                int3 neighbourGridPos = {gridPos.x + x, gridPos.y + y, gridPos.z + z};
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex = cellStart[neighbourGridIndex];
                // empty cell.
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i) {
                    float4 neighbour = predictedPos[i];
                    float3 r = {curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z};
                    density += wPoly6(r);
                    curGrad = wSpikyGrad(r);
                    curGrad.x *= params.m_invRestDensity;
                    curGrad.y *= params.m_invRestDensity;
                    curGrad.z *= params.m_invRestDensity;

                    gradSum_i.x += curGrad.x;
                    gradSum_i.y += curGrad.y;
                    gradSum_i.z += curGrad.z;
                    if (i != index)
                        gradSquaredSum_j += (curGrad.x * curGrad.x + curGrad.y * curGrad.y + curGrad.z * curGrad.z);
                }
            }
        }
    }
    gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;

    // density constraint.
    predictedPos[index].w = density;
    float constraint = density * params.m_invRestDensity - 1.0f;
    float lambda = -(constraint) / (gradSquaredSumTotal + params.m_lambdaEps);
    velocity[index] = {readVel.x, readVel.y, readVel.z, lambda};
}

__global__ void calculate_delta_positions(
    float4 *velocity,
    float4 *predictedPos,
    float3 *deltaPos,
    unsigned int *gridParticleHash,
    unsigned int *cellStart,
    unsigned int *cellEnd,
    unsigned int numParticles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float4 readPos = predictedPos[index];
    float4 readVel = velocity[index];
    float3 curPos = {readPos.x, readPos.y, readPos.z};
    int3 gridPos = calcGridPosKernel(curPos);

    float curLambda = readVel.w;
    float3 deltaP = {0.0f, 0.0f, 0.0f};
#pragma unroll 3
    for (int z = -1; z <= 1; ++z) {
#pragma unroll 3
        for (int y = -1; y <= 1; ++y) {
#pragma unroll 3
            for (int x = -1; x <= 1; ++x) {
                int3 neighbourGridPos = {gridPos.x + x, gridPos.y + y, gridPos.z + z};
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex = cellStart[neighbourGridIndex];
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i) {
                    float4 neighbour = predictedPos[i];
                    float neighbourLambda = velocity[i].w;
                    float3 r = {curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z};
                    float corrTerm = wPoly6(r) * params.m_oneDivWPoly6;
                    // float coff = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
                    float coff = curLambda + neighbourLambda;
                    float3 grad = wSpikyGrad(r);
                    deltaP.x += coff * grad.x;
                    deltaP.y += coff * grad.y;
                    deltaP.z += coff * grad.z;
                }
            }
        }
    }

    float3 ret = {deltaP.x * params.m_invRestDensity, deltaP.y * params.m_invRestDensity,
                  deltaP.z * params.m_invRestDensity};
    deltaPos[index] = ret;
}

__global__ void correct_predicted_positions(
    float4 *predictedPos,
    float3 *deltaPos,
    unsigned int numParticles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float4 readPos = predictedPos[index];
    float3 readDeltaPos = deltaPos[index];
    readDeltaPos.x = readPos.x + readDeltaPos.x;
    readDeltaPos.y = readPos.y + readDeltaPos.y;
    readDeltaPos.z = readPos.z + readDeltaPos.z;

    predictedPos[index] = {readDeltaPos.x, readDeltaPos.y, readDeltaPos.z, readPos.w};
}

#endif