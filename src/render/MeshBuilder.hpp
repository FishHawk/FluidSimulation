#ifndef MESH_BUILDER_HPP
#define MESH_BUILDER_HPP

#include "Mesh.hpp"

class Mesh3Builder {
private:
    std::vector<Vertex3> vertices_;

    Mesh3Builder& operator<<(Vertex3 vertex) {
        vertices_.push_back(vertex);
        return *this;
    };

public:
    enum class Direction {
        X_POSITIVE,
        X_NEGATIVE,
        Y_POSITIVE,
        Y_NEGATIVE,
        Z_POSITIVE,
        Z_NEGATIVE,
    };

    void add_triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 color);

    template <Direction dir>
    void add_surface(glm::vec3 p, glm::vec2 size, glm::vec3 color);

    void add_cube(glm::vec3 p, glm::vec3 size, glm::vec3 color);

    Mesh3* build_mesh() {
        return new Mesh3(vertices_);
    };
};

void Mesh3Builder::add_triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 color) {
    glm::vec3 norm = glm::normalize(glm::cross(p1 - p2, p1 - p3));
    vertices_.push_back(Vertex3{p1, color, norm});
    vertices_.push_back(Vertex3{p2, color, norm});
    vertices_.push_back(Vertex3{p3, color, norm});
};

template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::Y_POSITIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(0.0f, 1.0f, 0.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, size.y), color, norm});
}
template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::Y_NEGATIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(0.0f, -1.0f, 0.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, size.y), color, norm});
}
template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::X_POSITIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(1.0f, 0.0f, 0.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, size.y), color, norm});
}
template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::X_NEGATIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(-1.0f, 0.0f, 0.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, 0.0f, size.y), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.x, size.y), color, norm});
}
template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::Z_POSITIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(0.0f, 0.0f, 1.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.y, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.y, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, size.y, 0.0f), color, norm});
}
template <>
void Mesh3Builder::add_surface<Mesh3Builder::Direction::Z_NEGATIVE>(glm::vec3 p, glm::vec2 size, glm::vec3 color) {
    glm::vec3 norm = glm::vec3(0.0f, 0.0f, -1.0f);
    vertices_.push_back(Vertex3{p, color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.y, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, 0.0f, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(0.0f, size.y, 0.0f), color, norm});
    vertices_.push_back(Vertex3{p + glm::vec3(size.x, size.y, 0.0f), color, norm});
}

void Mesh3Builder::add_cube(glm::vec3 p, glm::vec3 size, glm::vec3 color) {
    add_surface<Direction::X_NEGATIVE>(p, glm::vec2(size.y, size.z), color);
    add_surface<Direction::Y_NEGATIVE>(p, glm::vec2(size.x, size.z), color);
    add_surface<Direction::Z_NEGATIVE>(p, glm::vec2(size.x, size.y), color);

    add_surface<Direction::X_POSITIVE>(p + glm::vec3(size.x, 0.0f, 0.0f), glm::vec2(size.y, size.z), color);
    add_surface<Direction::Y_POSITIVE>(p + glm::vec3(0.0f, size.y, 0.0f), glm::vec2(size.x, size.z), color);
    add_surface<Direction::Z_POSITIVE>(p + glm::vec3(0.0f, 0.0f, size.z), glm::vec2(size.x, size.y), color);
}
#endif