#ifndef BOX_CASE_HPP
#define BOX_CASE_HPP

#include "BaseCase.hpp"

class BoxCase : public BaseCase {
private:
    /* clang-format off */
    float box[216] = {
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,//
//
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,//
//
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,//
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,//
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,//
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,//
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,//
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,//
//
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,//
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,//
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,//
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,//
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,//
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,//
//
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,//
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,//
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,//
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,//
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,//
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,//
//
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,//
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,//
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f//
    };
    unsigned int box_vbo_, box_vao_;
    Shader box_shader_;

    float plane[36] = {
        -0.5f,  0.0f, -0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.0f, -0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.0f,  0.5f,  0.0f,  1.0f,  0.0f,//
         0.5f,  0.0f,  0.5f,  0.0f,  1.0f,  0.0f,//
        -0.5f,  0.0f,  0.5f,  0.0f,  1.0f,  0.0f,//
        -0.5f,  0.0f, -0.5f,  0.0f,  1.0f,  0.0f//
    };
    unsigned int plane_vbo_, plane_vao_;
    Shader plane_shader_;

public:
    BoxCase();
    ~BoxCase();

    void render();
};

BoxCase::BoxCase()
    : box_shader_("src/glsl/box.vs", "src/glsl/box.fs"),
      plane_shader_("src/glsl/plane.vs", "src/glsl/plane.fs") {
    // for box
    glGenVertexArrays(1, &box_vao_);
    glGenBuffers(1, &box_vbo_);

    glBindVertexArray(box_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, box_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(box), box, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // for plane
    glGenVertexArrays(1, &plane_vao_);
    glGenBuffers(1, &plane_vbo_);

    glBindVertexArray(plane_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, plane_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(plane), plane, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
};

BoxCase::~BoxCase() {}

void BoxCase::render() {
    glm::mat4 model;
    glm::mat4 view = camera_.get_view_matrix();
    glm::mat4 projection = camera_.get_projection_matrix();

    box_shader_.use();
    glBindVertexArray(box_vao_);

    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.5f, 0.0f));
    // model = glm::rotate(model, glm::radians(0.f), glm::vec3(1.0f, 0.3f, 0.5f));
    box_shader_.set_uniform("model", model);
    box_shader_.set_uniform("view", view);
    box_shader_.set_uniform("projection", projection);
    box_shader_.set_uniform("view_pos", camera_.get_position());
    glDrawArrays(GL_TRIANGLES, 0, 36);

    plane_shader_.use();
    glBindVertexArray(plane_vao_);

    model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(30.0f, 1.0f, 30.0f));
    plane_shader_.set_uniform("model", model);
    plane_shader_.set_uniform("view", view);
    plane_shader_.set_uniform("projection", projection);
    glDrawArrays(GL_TRIANGLES, 0, 6);
};

#endif