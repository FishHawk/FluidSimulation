#include <iostream>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // include glfw3.h after opengl definitions

#include "Input.hpp"
#include "SceneBuilder.hpp"
#include "Ui.hpp"

static GLFWwindow *window;

int main(int argc, char *argv[]) {
    // initialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create window
    window = glfwCreateWindow(1400, 1000, "FluidSimulation", NULL, NULL);
    if (window == nullptr) {
        std::cout << "Error: Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // load all opengl function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Error: Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // build scene
    std::string device = "cpu";
    if (argc > 1)
        device = argv[1];
    auto [render_system, simulate_system] = SceneBuilder::build_scene(device);

    // initialize input
    Input::link_to_systems(render_system, simulate_system);
    Input::register_callback(window);

    // initialize ui
    Ui::link_to_systems(render_system, simulate_system);
    Ui::init(window);

    // main loop
    while (!glfwWindowShouldClose(window)) {
        // clear
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // render world
        render_system.update_particles(simulate_system.get_particle_position());
        render_system.render();

        // render ui
        Ui::render();

        // swap buffers and poll io events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // terminate
    simulate_system.terminate();
    glfwTerminate();
    return 0;
}
