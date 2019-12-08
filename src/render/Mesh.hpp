#ifndef RENDER_MESH_HPP
#define RENDER_MESH_HPP

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace render {

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

} // namespace render

#endif