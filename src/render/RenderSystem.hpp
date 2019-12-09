#ifndef RENDER_RENDER_SYSTEM_HPP
#define RENDER_RENDER_SYSTEM_HPP

#include <map>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h> // include glfw3.h after opengl definitions

#include "Drawable.hpp"
#include "MeshBuilder.hpp"
#include "Shader.hpp"
#include "camera/ThirdPersonCamera.hpp"

namespace render {

class RenderSystem {
private:
    std::map<std::string, Mesh *> mesh_manager_;
    std::map<std::string, Shader *> shader_manager_;
    std::map<std::string, Drawable *> drawable_manager_;

    ThirdPersonCamera camera_;

    // fluid
    double particle_radius_ = 0.025;

    // axis
    bool is_axis_enabled_ = false;

    // container
    bool is_container_enabled_ = false;
    glm::vec3 container_position_ = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 container_size_ = glm::vec3(1.0f, 1.0f, 1.0f);

    RenderSystem();
    RenderSystem(RenderSystem const &) = delete;
    void operator=(RenderSystem const &) = delete;

public:
    const int defuault_window_width = 1400;
    const int defuault_window_height = 1000;

    static RenderSystem &get_instance() {
        static RenderSystem instance;
        return instance;
    };
    ~RenderSystem();

    // fluid
    void set_particle_radius(double radius) { particle_radius_ = radius; }
    void update_particles(std::vector<glm::vec3> positions);

    // axis
    bool *get_axis_switch() { return &is_axis_enabled_; };

    // container
    bool *get_container_switch() { return &is_container_enabled_; };
    void set_container(glm::vec3 container_position, glm::vec3 container_size) {
        container_position_ = container_position;
        container_size_ = container_size;
    }

    // render
    ThirdPersonCamera &get_camera() { return camera_; }
    void render();
};

} // namespace render

#endif