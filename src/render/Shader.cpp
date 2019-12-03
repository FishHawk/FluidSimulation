#include "Shader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace render;

void check_compile_errors(unsigned int shader, std::string type) {
    int success;
    char infoLog[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                  << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
    }
}

void check_link_errors(unsigned int shader) {
    int success;
    char infoLog[1024];
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader, 1024, NULL, infoLog);
        std::cout << "ERROR::PROGRAM_LINKING_ERROR\n"
                  << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
    }
}

std::string read_glsl_code(const std::string file_path) {
    std::ifstream file;
    file.open(file_path);

    std::stringstream file_stream;
    file_stream << file.rdbuf();
    return file_stream.str();
}

Shader::Shader(const std::string vertex_path, const std::string fragment_path) {
    // retrieve the vertex/fragment source code from filepath
    std::string vertex_string = read_glsl_code(vertex_path),
                fragment_string = read_glsl_code(fragment_path);

    const char* vertex_code = vertex_string.c_str();
    const char* fragment_code = fragment_string.c_str();

    // compile vertex shader
    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertex_code, NULL);
    glCompileShader(vertex);
    check_compile_errors(vertex, "VERTEX");

    // compile fragment shader
    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragment_code, NULL);
    glCompileShader(fragment);
    check_compile_errors(fragment, "FRAGMENT");

    // link program
    id_ = glCreateProgram();
    glAttachShader(id_, vertex);
    glAttachShader(id_, fragment);
    glLinkProgram(id_);
    check_link_errors(id_);

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}
