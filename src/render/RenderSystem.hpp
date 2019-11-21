#ifndef RENDER_SYSTEM_HPP
#define RENDER_SYSTEM_HPP

#include <glad/glad.h>

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

public:
    const int defuault_window_width = 1400;
    const int defuault_window_height = 1000;

    RenderSystem();
    ~RenderSystem();

    void init_particals();

    void render();

    void update_particles(std::vector<glm::vec3> positions);

    void process_keyboard_input(GLFWwindow *window, float delta_time) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::FORWARD, delta_time);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::BACKWARD, delta_time);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::LEFT, delta_time);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::RIGHT, delta_time);
    }

    void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);
        camera_.change_frame_ratio((float)width / (float)height);
    }

    void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
        static float xlast = 0;
        static float ylast = 0;
        static bool is_first = true;

        if (is_first) {
            xlast = xpos;
            ylast = ypos;
            is_first = false;
        }

        float xoffset = xpos - xlast;
        float yoffset = ylast - ypos;
        camera_.rotate(xoffset, yoffset);

        xlast = xpos;
        ylast = ypos;
    }

    void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
        camera_.zoom(yoffset);
    }
};

RenderSystem::RenderSystem()
    : camera_(glm::vec3(-3.0f, 2.0f, 0.0f), 0.0f, 0.0f, 1.4f) {
    shader_manager_["simple3"] = new Shader("src/glsl/simple3.vs", "src/glsl/simple3.fs");
    shader_manager_["particals"] = new Shader("src/glsl/particals.vs", "src/glsl/simple3.fs");

    {
        Mesh3Builder builder;
        for (int x = -30; x < 30; x++) {
            for (int z = -30; z < 30; z++) {
                glm::vec3 color(1.0f, 1.0f, 1.0f);
                if ((x + z) % 2 == 0)
                    color = glm::vec3(0.7f, 0.7f, 0.7f);
                builder.add_surface<Mesh3Builder::Direction::Y_POSITIVE>(
                    glm::vec3(float(x), 0.0f, float(z)),
                    glm::vec2(1.0f, 1.0f),
                    color);
            }
        }
        auto mesh = builder.build_mesh();
        mesh_manager_["floor"] = mesh;
        drawable_manager_["floor"] = new Drawable3(mesh);
    }
    {
        Mesh3Builder builder;
        builder.add_cube(
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 0.0f));
        auto mesh = builder.build_mesh();
        mesh_manager_["box"] = mesh;
        drawable_manager_["box"] = new Drawable3(mesh);
    }
    {
        Mesh3Builder builder;
        builder.add_icosphere(
            glm::vec3(0.0f, 0.0f, 0.0f),
            0.5,
            glm::vec3(0.0f, 1.0f, 1.0f),
            3);
        auto mesh = builder.build_mesh();
        mesh_manager_["partical"] = mesh;
        drawable_manager_["particals"] = new InstanceDrawable3(mesh);
    }
    {
        Mesh3Builder builder;
        builder.add_icosphere(
            glm::vec3(0.0f, 0.0f, 0.0f),
            0.5,
            glm::vec3(0.0f, 1.0f, 1.0f));
        auto mesh = builder.build_mesh();
        mesh_manager_["sphere"] = mesh;
        drawable_manager_["sphere"] = new Drawable3(mesh);
    }
}

RenderSystem::~RenderSystem() {}

void RenderSystem::update_particles(std::vector<glm::vec3> positions) {
    dynamic_cast<InstanceDrawable3 *>(drawable_manager_["particals"])->update_positions(positions);
}

void RenderSystem::render() {
    glm::mat4 model;
    glm::mat4 view = camera_.view_matrix();
    glm::mat4 projection = camera_.projection_matrix();

    auto &simple3_shader_ = *shader_manager_["simple3"];
    simple3_shader_.use();
    simple3_shader_.set_uniform("view", view);
    simple3_shader_.set_uniform("projection", projection);
    simple3_shader_.set_uniform("view_pos", camera_.position());

    model = glm::mat4(1.0f);
    simple3_shader_.set_uniform("model", model);
    drawable_manager_["floor"]->draw();
    // drawable_manager_["box"]->draw();
    // drawable_manager_["sphere"]->draw();

    auto &particals_shader_ = *shader_manager_["particals"];
    particals_shader_.use();
    particals_shader_.set_uniform("view", view);
    particals_shader_.set_uniform("projection", projection);
    particals_shader_.set_uniform("view_pos", camera_.position());
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.025f));
    particals_shader_.set_uniform("model", model);
    drawable_manager_["particals"]->draw();
};

#endif