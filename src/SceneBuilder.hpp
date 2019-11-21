#ifndef SCENE_BUILDER
#define SCENE_BUILDER

#include <glm/glm.hpp>
#include <vector>

class SceneBuilder {
public:
    static std::vector<glm::vec3> init_fluid_particles();
    static std::vector<glm::vec3> init_boundary_particles();
};

#endif