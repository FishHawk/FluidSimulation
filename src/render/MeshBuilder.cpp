#include "MeshBuilder.hpp"

using namespace render;

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
    add_surface<Direction::X_NEGATIVE>(p - size / 2.0f, glm::vec2(size.y, size.z), color);
    add_surface<Direction::Y_NEGATIVE>(p - size / 2.0f, glm::vec2(size.x, size.z), color);
    add_surface<Direction::Z_NEGATIVE>(p - size / 2.0f, glm::vec2(size.x, size.y), color);

    add_surface<Direction::X_POSITIVE>(p - size / 2.0f + glm::vec3(size.x, 0.0f, 0.0f), glm::vec2(size.y, size.z), color);
    add_surface<Direction::Y_POSITIVE>(p - size / 2.0f + glm::vec3(0.0f, size.y, 0.0f), glm::vec2(size.x, size.z), color);
    add_surface<Direction::Z_POSITIVE>(p - size / 2.0f + glm::vec3(0.0f, 0.0f, size.z), glm::vec2(size.x, size.y), color);
}

void Mesh3Builder::add_icosphere(glm::vec3 p, float r, glm::vec3 color, unsigned int level) {
    std::vector<glm::vec3> points;
    std::vector<glm::ivec3> indices;

    // create 12 vertices of a icosahedron
    float f = 1.0 / 1.9021 * r;
    float t = (1.0 + sqrt(5.0)) / 2.0 / 1.9021 * r;

    points.push_back(glm::vec3(-f, t, 0.0f));
    points.push_back(glm::vec3(f, t, 0.0f));
    points.push_back(glm::vec3(-f, -t, 0.0f));
    points.push_back(glm::vec3(f, -t, 0.0f));

    points.push_back(glm::vec3(0.0f, -f, t));
    points.push_back(glm::vec3(0.0f, f, t));
    points.push_back(glm::vec3(0.0f, -f, -t));
    points.push_back(glm::vec3(0.0f, f, -t));

    points.push_back(glm::vec3(t, 0.0f, -f));
    points.push_back(glm::vec3(t, 0.0f, f));
    points.push_back(glm::vec3(-t, 0.0f, -f));
    points.push_back(glm::vec3(-t, 0.0f, f));

    // 5 faces around point 0
    indices.push_back(glm::ivec3(0, 11, 5));
    indices.push_back(glm::ivec3(0, 5, 1));
    indices.push_back(glm::ivec3(0, 1, 7));
    indices.push_back(glm::ivec3(0, 7, 10));
    indices.push_back(glm::ivec3(0, 10, 11));

    // 5 adjacent faces
    indices.push_back(glm::ivec3(1, 5, 9));
    indices.push_back(glm::ivec3(5, 11, 4));
    indices.push_back(glm::ivec3(11, 10, 2));
    indices.push_back(glm::ivec3(10, 7, 6));
    indices.push_back(glm::ivec3(7, 1, 8));

    // 5 faces around point 3
    indices.push_back(glm::ivec3(3, 9, 4));
    indices.push_back(glm::ivec3(3, 4, 2));
    indices.push_back(glm::ivec3(3, 2, 6));
    indices.push_back(glm::ivec3(3, 6, 8));
    indices.push_back(glm::ivec3(3, 8, 9));

    // 5 adjacent faces
    indices.push_back(glm::ivec3(4, 9, 5));
    indices.push_back(glm::ivec3(2, 4, 11));
    indices.push_back(glm::ivec3(6, 2, 10));
    indices.push_back(glm::ivec3(8, 6, 7));
    indices.push_back(glm::ivec3(9, 8, 1));

    // refine triangles
    for (int i = 0; i < level; i++) {
        std::vector<glm::ivec3> new_indices;
        for (auto &indice : indices) {
            // replace triangle by 4 triangles
            auto i = points.size();
            points.push_back(r * glm::normalize(points[indice[0]] + points[indice[1]]));
            points.push_back(r * glm::normalize(points[indice[1]] + points[indice[2]]));
            points.push_back(r * glm::normalize(points[indice[2]] + points[indice[0]]));

            new_indices.push_back(glm::ivec3(i, i + 1, i + 2));
            new_indices.push_back(glm::ivec3(indice[0], i, i + 2));
            new_indices.push_back(glm::ivec3(indice[1], i + 1, i));
            new_indices.push_back(glm::ivec3(indice[2], i + 2, i + 1));
        }
        indices = new_indices;
    }

    // build mesh
    for (auto &indice : indices) {
        add_triangle(points[indice[0]], points[indice[1]], points[indice[2]], color);
    }
}