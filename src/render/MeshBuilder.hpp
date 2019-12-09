#ifndef RENDER_MESH_BUILDER_HPP
#define RENDER_MESH_BUILDER_HPP

#include "Mesh.hpp"

namespace render {

class Mesh3Builder {
private:
    std::vector<Vertex3> vertices_;

    Mesh3Builder &operator<<(Vertex3 vertex) {
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

    Mesh3Builder&  add_triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 color);
    template <Direction dir>
    Mesh3Builder&  add_surface(glm::vec3 p, glm::vec2 size, glm::vec3 color);
    Mesh3Builder&  add_cube(glm::vec3 p, glm::vec3 size, glm::vec3 color);
    Mesh3Builder&  add_icosphere(glm::vec3 p, float r, glm::vec3 color, unsigned int level = 3);
    // void add_uvsphere(glm::vec3 p, float r, glm::vec3 color);

    Mesh3 *build_mesh() {
        return new Mesh3(vertices_);
    };
};

class Mesh2Builder {
private:
    std::vector<Vertex2> vertices_;

    Mesh2Builder &operator<<(Vertex2 vertex) {
        vertices_.push_back(vertex);
        return *this;
    };

public:
    Mesh2Builder& add_line(glm::vec3 p1, glm::vec3 p2, glm::vec3 color);
    Mesh2Builder& add_cube_frame(glm::vec3 p, glm::vec3 size, glm::vec3 color);

    Mesh2 *build_mesh() {
        return new Mesh2(vertices_);
    };
};

} // namespace render

#endif