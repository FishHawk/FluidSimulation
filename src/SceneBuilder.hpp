#ifndef SCENE_BUILDER
#define SCENE_BUILDER

#include <glm/glm.hpp>
#include <vector>

#include "render/RenderSystem.hpp"
#include "simulation/FluidSolver.hpp"

class SceneBuilder {
private:
    static std::vector<glm::vec3> init_fluid_particles(double particle_radius);
    static std::vector<glm::vec3> init_boundary_particles(double particle_radius);

public:
    static std::pair<RenderSystem &, Simulation::FluidSolver &> build_scene(std::string scene_name);
};

#endif