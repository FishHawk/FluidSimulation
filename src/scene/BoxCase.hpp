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


    unsigned int partical_vbo_, partical_vao_;
    unsigned int partical_instance_vbo_;
    Shader partical_shader_;
    /* clang-format on */
public:
    BoxCase();
    ~BoxCase();

    void init_box();
    void init_plane();
    void init_particals();

    void render();
};

BoxCase::BoxCase()
    : box_shader_("src/glsl/box.vs", "src/glsl/box.fs"),
      plane_shader_("src/glsl/plane.vs", "src/glsl/plane.fs"),
      partical_shader_("src/glsl/partical.vs", "src/glsl/partical.fs") {
    init_box();
    init_plane();
    init_particals();
};

BoxCase::~BoxCase() {}

void BoxCase::init_box() {
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
}

void BoxCase::init_plane() {
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
}

void BoxCase::init_particals() {
    glGenVertexArrays(1, &partical_vao_);
    // glGenBuffers(1, &partical_vbo_);

    glBindVertexArray(partical_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, box_vbo_);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(box), box, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    static glm::vec3 translations[1000];
    int index = 0;
    for (int z = 0; z < 10; z += 1) {
        for (int y = 0; y < 10; y += 1) {
            for (int x = 0; x < 10; x += 1) {
                translations[index++] = glm::vec3((float)x * 0.1, (float)y * 0.1, (float)z * 0.1);
            }
        }
    }
    glGenBuffers(1, &partical_instance_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, partical_instance_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 1000, &translations[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(2, 11);
}

void BoxCase::render() {
    glm::mat4 model;
    glm::mat4 view = camera_.view_matrix();
    glm::mat4 projection = camera_.projection_matrix();

    // render box
    box_shader_.use();
    glBindVertexArray(box_vao_);

    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.5f, 0.0f));
    // model = glm::rotate(model, glm::radians(0.f), glm::vec3(1.0f, 0.3f, 0.5f));
    box_shader_.set_uniform("model", model);
    box_shader_.set_uniform("view", view);
    box_shader_.set_uniform("projection", projection);
    box_shader_.set_uniform("view_pos", camera_.position());
    // glDrawArrays(GL_TRIANGLES, 0, 36);

    // render plane
    plane_shader_.use();
    glBindVertexArray(plane_vao_);

    model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(30.0f, 1.0f, 30.0f));
    plane_shader_.set_uniform("model", model);
    plane_shader_.set_uniform("view", view);
    plane_shader_.set_uniform("projection", projection);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // render particals
    partical_shader_.use();
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.08f, 0.08f, 0.08f));
    box_shader_.set_uniform("model", model);
    box_shader_.set_uniform("view", view);
    box_shader_.set_uniform("projection", projection);
    box_shader_.set_uniform("view_pos", camera_.position());
    glBindVertexArray(partical_vao_);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, 36000);
};

#endif