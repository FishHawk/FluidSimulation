#include "Input.hpp"

render::RenderSystem *Input::render_system_ = nullptr;
simulate::SimulateSystem *Input::simulate_system_ = nullptr;

void Input::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    render_system_->get_camera().change_frame_ratio((float)width / (float)height);
}

void Input::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;

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
    render_system_->get_camera().rotate(xoffset, yoffset);

    xlast = xpos;
    ylast = ypos;
}

void Input::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    auto &render_system = render::RenderSystem::get_instance();
    render_system_->get_camera().slide(yoffset);
}

void Input::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        if (simulate_system_->is_running())
            simulate_system_->stop();
        else
            simulate_system_->start();
    }
}