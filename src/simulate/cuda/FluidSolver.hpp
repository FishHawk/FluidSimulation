#ifndef SIMULATE_CUDA_FLUID_SOLVER_CUH
#define SIMULATE_CUDA_FLUID_SOLVER_CUH

#include <vector_types.h>

namespace simulate {
namespace cuda {

// helper
void print_gpu_info();
void check_cuda_error(const char *errorMessage);

class FluidSolverCuda {
public:
    struct Parameters {
        float3 container_start, container_end;
        float particle_radius;
        float sph_radius, sph_radius_2;
        float poly6_coff, spiky_grad_coff;
        float density, inv_density;
        uint3 grid_size;
    };

    float4 *positions;
    float4 *velocities;
    float4 *predicted_positions;

    unsigned int *cell_ids;
    unsigned int *cell_start;
    unsigned int *cell_end;

    unsigned int particle_number;
    unsigned int cell_number;

    static void set_parameters(Parameters& params);

    void malloc();
    void simulate(float delta_time);
};

} // namespace cuda
} // namespace simulate


#endif