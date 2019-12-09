#include "RenderSystem.hpp"

#include <iostream>
using namespace render;

RenderSystem::RenderSystem()
    : camera_(glm::vec3(0.0f, 0.0f, 0.0f), 0.0f, -30.0f, 1.4f, 10.0f) {
    shader_manager_["simple2"] = new Shader("src/glsl/simple2.vs", "src/glsl/simple2.fs");
    shader_manager_["simple3"] = new Shader("src/glsl/simple3.vs", "src/glsl/simple3.fs");
    shader_manager_["particals"] = new Shader("src/glsl/particals.vs", "src/glsl/simple3.fs");

    {
        Mesh3Builder builder;
        for (int x = -30; x < 30; x++) {
            for (int z = -30; z < 30; z++) {
                glm::vec3 color(1.0f, 1.0f, 1.0f);
                if ((x + z) % 2 == 0)
                    color = glm::vec3(0.7f, 0.7f, 0.7f);
                builder.add_surface<Mesh3Builder::Direction::Y_POSITIVE>(glm::vec3(float(x), 0.0f, float(z)), glm::vec2(1.0f, 1.0f), color);
            }
        }
        auto mesh = builder.build_mesh();
        mesh_manager_["floor"] = mesh;
        drawable_manager_["floor"] = new Drawable3(mesh);
    }
    {
        Mesh3Builder builder;
        builder.add_icosphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0, glm::vec3(0.0f, 1.0f, 1.0f), 3);
        auto mesh = builder.build_mesh();
        mesh_manager_["partical"] = mesh;
        drawable_manager_["particals"] = new InstanceDrawable3(mesh);
    }
    {
        Mesh2Builder builder;
        builder.add_line(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(30.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f))
            .add_line(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 30.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f))
            .add_line(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 30.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        auto mesh = builder.build_mesh();
        mesh_manager_["axis"] = mesh;
        drawable_manager_["axis"] = new Drawable2(mesh);
    }
    {
        const glm::vec3 container_color{0.0f, 0.0f, 0.0f};
        Mesh2Builder builder;
        auto mesh = builder.add_cube_frame(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), container_color).build_mesh();
        mesh_manager_["container"] = mesh;
        drawable_manager_["container"] = new Drawable2(mesh);
    }
}

RenderSystem::~RenderSystem() {}

void RenderSystem::update_particles(std::vector<glm::vec3> positions) {
    dynamic_cast<InstanceDrawable3 *>(drawable_manager_["particals"])->update_positions(positions);
}

void RenderSystem::render() {
    // enable opengl capabilities
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_CULL_FACE);

    glm::mat4 model;
    glm::mat4 view = camera_.view_matrix();
    glm::mat4 projection = camera_.projection_matrix();

    {
        auto &simple3_shader_ = *shader_manager_["simple3"];
        simple3_shader_.use();
        simple3_shader_.set_uniform("view", view);
        simple3_shader_.set_uniform("projection", projection);
        simple3_shader_.set_uniform("view_pos", camera_.position());

        model = glm::mat4(1.0f);
        simple3_shader_.set_uniform("model", model);
        glDisable(GL_DEPTH_TEST);
        drawable_manager_["floor"]->draw();
        glEnable(GL_DEPTH_TEST);
    }
    {
        auto &simple2_shader_ = *shader_manager_["simple2"];
        simple2_shader_.use();
        simple2_shader_.set_uniform("view", view);
        simple2_shader_.set_uniform("projection", projection);

        if (is_axis_enabled_) {
            model = glm::mat4(1.0f);
            simple2_shader_.set_uniform("model", model);
            drawable_manager_["axis"]->draw();
        }

        if (is_container_enabled_) {
            model = glm::mat4(1.0f);
            model = glm::translate(model, container_position_);
            model = glm::scale(model, container_size_);
            simple2_shader_.set_uniform("model", model);
            drawable_manager_["container"]->draw();
        }
    }
    {
        auto &particals_shader_ = *shader_manager_["particals"];
        particals_shader_.use();
        particals_shader_.set_uniform("view", view);
        particals_shader_.set_uniform("projection", projection);
        particals_shader_.set_uniform("view_pos", camera_.position());

        model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(particle_radius_ * 2));
        particals_shader_.set_uniform("model", model);
        drawable_manager_["particals"]->draw();
    }
};
