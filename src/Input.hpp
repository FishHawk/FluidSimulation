#ifndef INPUT_HPP
#define INPUT_HPP

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // include glfw3.h after opengl definitions

#include "render/RenderSystem.hpp"

class Input {
public:
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        auto &render_system = render::RenderSystem::get_instance();
        glViewport(0, 0, width, height);
        render_system.get_camera().change_frame_ratio((float)width / (float)height);
    }

    static void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
        if (ImGui::GetIO().WantCaptureMouse)
            return;

        auto &render_system = render::RenderSystem::get_instance();
        static float xlast = 0;
        static float ylast = 0;
        static bool is_first = true;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            is_first = true;
            return;
        }

        if (is_first) {
            xlast = xpos;
            ylast = ypos;
            is_first = false;
        }

        float xoffset = xpos - xlast;
        float yoffset = ylast - ypos;
        render_system.get_camera().rotate(xoffset, yoffset);

        xlast = xpos;
        ylast = ypos;
    }

    static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
        if (ImGui::GetIO().WantCaptureMouse)
            return;

        auto &render_system = render::RenderSystem::get_instance();
        render_system.get_camera().slide(yoffset);
    }

    static void process_keyboard_input(GLFWwindow *window, float delta_time) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }
};

#endif