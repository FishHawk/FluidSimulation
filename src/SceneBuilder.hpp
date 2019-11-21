#ifndef SCENE_BUILDER
#define SCENE_BUILDER

#include <glm/glm.hpp>
#include <vector>

#include "render/RenderSystem.hpp"
#include "simulation/FluidSolver.hpp"

class SceneBuilder {
private:
    static std::vector<glm::vec3> init_fluid_particles();
    static std::vector<glm::vec3> init_boundary_particles();

public:
    static void build_scene(RenderSystem& render_system, FluidSolver& fluid_solver);
};

#endif