#ifndef INPUT_HPP
#define INPUT_HPP

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "render/RenderSystem.hpp"
#include "simulate/SimulateSystem.hpp"

class Input {
private:
    static render::RenderSystem *render_system_;
    static simulate::SimulateSystem *simulate_system_;

    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
    static void mouse_callback(GLFWwindow *window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);

public:
    static void link_to_systems(render::RenderSystem &render_system, simulate::SimulateSystem &simulate_system) {
        render_system_ = &render_system;
        simulate_system_ = &simulate_system;
    }

    static void register_callback(GLFWwindow *window) {
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetKeyCallback(window, key_callback);
    }
};

#endif