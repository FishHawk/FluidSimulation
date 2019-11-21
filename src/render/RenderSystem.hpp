#ifndef RENDER_SYSTEM_HPP
#define RENDER_SYSTEM_HPP

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <map>

#include "Drawable.hpp"
#include "FpsCamera.hpp"
#include "MeshBuilder.hpp"
#include "Shader.hpp"

class RenderSystem {
private:
    std::map<std::string, Mesh *> mesh_manager_;
    std::map<std::string, Shader *> shader_manager_;
    std::map<std::string, Drawable *> drawable_manager_;

    FpsCamera camera_;

    double particle_radius_ = 0.025;

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

    // config
    void set_particle_radius(double radius) { particle_radius_ = radius; }

    // input
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
    static void mouse_callback(GLFWwindow *window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
    void process_keyboard_input(GLFWwindow *window, float delta_time);

    void render();

    void update_particles(std::vector<glm::vec3> positions);
};

#endif