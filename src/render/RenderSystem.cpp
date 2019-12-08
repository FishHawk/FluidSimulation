#include "RenderSystem.hpp"

#include <iostream>
using namespace render;

void RenderSystem::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    auto &render_system = get_instance();
    glViewport(0, 0, width, height);
    render_system.camera_.change_frame_ratio((float)width / (float)height);
}

void RenderSystem::mouse_callback(GLFWwindow *window, double xpos, double ypos) {

    auto &render_system = get_instance();
    static float xlast = 0;
    static float ylast = 0;
    static bool is_first = true;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
        is_first = true;
        return;
    }

    if (is_first) {
        xlast = xpos;
        ylast = ypos;
        is_first = false;
    }

    float xoffset = xpos - xlast;
    float yoffset = ylast - ypos;
    render_system.camera_.rotate(xoffset, yoffset);

    xlast = xpos;
    ylast = ypos;
}

void RenderSystem::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    auto &render_system = get_instance();
    render_system.camera_.slide(yoffset);
}

void RenderSystem::process_keyboard_input(GLFWwindow *window, float delta_time) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

RenderSystem::RenderSystem()
    : camera_(glm::vec3(0.0f, 0.0f, 0.0f), 0.0f, -30.0f, 1.4f, 10.0f) {
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
            1.0,
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
    model = glm::scale(model, glm::vec3(particle_radius_ * 2));
    particals_shader_.set_uniform("model", model);
    drawable_manager_["particals"]->draw();
};
