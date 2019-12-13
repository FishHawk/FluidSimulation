#ifndef SCENE_BUILDER
#define SCENE_BUILDER

#include "render/RenderSystem.hpp"
#include "simulate/SimulateSystem.hpp"

class SceneBuilder {
private:
    static simulate::SimulateSystem &get_simulate_system(std::string device);

public:
    static std::pair<render::RenderSystem &, simulate::SimulateSystem &> build_scene(std::string device);
};

#endif