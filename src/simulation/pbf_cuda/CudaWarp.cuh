#ifndef CUDA_WARP_CUH
#define CUDA_WARP_CUH

#include <vector_types.h>

namespace Simulation {
namespace PbfCuda {

class FluidSolverCuda {
public:
    float4 *positions;
    float4 *velocities;
    float4 *predicted_positions;
    float3 *delta_positions;

    unsigned int *cell_ids;
    unsigned int *cell_start;
    unsigned int *cell_end;

    unsigned int particles_number;

    void malloc();
    void simulate(float delta_time);
};

} // namespace PbfCuda
} // namespace Simulation

struct SimulateParams {
    uint3 m_gridSize;
    float3 m_cellSize;
    float3 m_worldOrigin;
    float m_poly6Coff;
    float m_spikyGradCoff;
    float m_sphRadius;
    float m_sphRadiusSquared;
    float m_lambdaEps;
    float m_restDensity;
    float m_invRestDensity;
    float m_particleRadius;
    float m_oneDivWPoly6;
    unsigned int m_numGridCells;
};
void setParameters(SimulateParams *hostParams);
void print_gpu_info();
void check_cuda_error(const char *errorMessage);

#endif