#ifndef BASE_CASE_HPP
#define BASE_CASE_HPP

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../render/FpsCamera.hpp"
#include "../render/Shader.hpp"

class BaseCase {
protected:
    FpsCamera camera_;

public:
    const int defuault_window_width = 1400;
    const int defuault_window_height = 1000;

    BaseCase();
    ~BaseCase();

    virtual void render() = 0;

    void process_keyboard_input(GLFWwindow* window, float delta_time) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::FORWARD, delta_time);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::BACKWARD, delta_time);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::LEFT, delta_time);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera_.move(Camera::MovementDirection::RIGHT, delta_time);
    }

    void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        camera_.change_frame_ratio((float)width / (float)height);
    }

    void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        static float xlast = 0;
        static float ylast = 0;
        static bool is_first = true;

        if (is_first) {
            xlast = xpos;
            ylast = ypos;
            is_first = false;
        }

        float xoffset = xpos - xlast;
        float yoffset = ylast - ypos;
        camera_.rotate(xoffset, yoffset);

        xlast = xpos;
        ylast = ypos;
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        camera_.zoom(yoffset);
    }
};

BaseCase::BaseCase() : camera_(glm::vec3(-3.0f, 2.0f, 0.0f), 0.0f, 0.0f, 1.4f) {}

BaseCase::~BaseCase() {}

#endif