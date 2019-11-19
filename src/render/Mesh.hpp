#ifndef MESH_HPP
#define MESH_HPP

#include <glm/glm.hpp>
#include <vector>

struct Vertex3 {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal;
};

struct Vertex2 {
    glm::vec2 position;
    glm::vec3 color;
};

class Mesh {
protected:
    unsigned int vao_, vbo_;
    size_t size_;

public:
    size_t size() const { return size_; };
    unsigned int id() const { return vao_; };
    void bind() { glBindVertexArray(vao_); };
    void unbind() { glBindVertexArray(0); };
};

class Mesh3 : public Mesh {
public:
    Mesh3(std::vector<Vertex3> vertices);
};

Mesh3::Mesh3(std::vector<Vertex3> vertices) {
    size_ = vertices.size();

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex3), vertices.data(), GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // normal attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

#endif