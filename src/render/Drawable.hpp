#ifndef DRAWABLE_HPP
#define DRAWABLE_HPP

#include "Mesh.hpp"

class Drawable {
public:
    virtual void draw() = 0;
};

class Drawable3 : public Drawable {
private:
    Mesh3* mesh_;

public:
    Drawable3(Mesh3* mesh) : mesh_(mesh){};
    void draw() {
        mesh_->bind();
        glDrawArrays(GL_TRIANGLES, 0, mesh_->size());
        mesh_->unbind();
    };
};

class InstanceDrawable3 : public Drawable {
private:
    Mesh3* mesh_;
    unsigned int vbo_;
    std::vector<glm::vec3> positions_;

public:
    InstanceDrawable3(Mesh3* mesh) : mesh_(mesh) {};

    void update_positions(std::vector<glm::vec3>& positions) {
        positions_ = positions;
        auto size = positions.size();

        mesh_->bind();
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(glm::vec3), &positions_[0], GL_STATIC_DRAW);

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(3);
        glVertexAttribDivisor(3, 1);
    }

    void draw() {
        mesh_->bind();
        glDrawArraysInstanced(GL_TRIANGLES, 0, mesh_->size(), positions_.size());
        mesh_->unbind();
    };
};
#endif