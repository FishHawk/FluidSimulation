#ifndef RENDER_SHADER_HPP
#define RENDER_SHADER_HPP

#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace render {

class Shader {
public:
    Shader(const std::string vertex_path, const std::string fragment_path);

    void use() {
        glUseProgram(id_);
    }

    void set_uniform(const std::string &name, bool value) const {
        glUniform1i(glGetUniformLocation(id_, name.c_str()), (int)value);
    }
    void set_uniform(const std::string &name, int value) const {
        glUniform1i(glGetUniformLocation(id_, name.c_str()), value);
    }
    void set_uniform(const std::string &name, float value) const {
        glUniform1f(glGetUniformLocation(id_, name.c_str()), value);
    }

    void set_uniform(const std::string &name, const glm::vec2 &value) const {
        glUniform2fv(glGetUniformLocation(id_, name.c_str()), 1, &value[0]);
    }
    void set_uniform(const std::string &name, const glm::vec3 &value) const {
        glUniform3fv(glGetUniformLocation(id_, name.c_str()), 1, &value[0]);
    }
    void set_uniform(const std::string &name, const glm::vec4 &value) const {
        glUniform4fv(glGetUniformLocation(id_, name.c_str()), 1, &value[0]);
    }

    void set_uniform(const std::string &name, const glm::mat2 &mat) const {
        glUniformMatrix2fv(glGetUniformLocation(id_, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void set_uniform(const std::string &name, const glm::mat3 &mat) const {
        glUniformMatrix3fv(glGetUniformLocation(id_, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void set_uniform(const std::string &name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(id_, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

private:
    unsigned int id_;
};

} // namespace render

#endif