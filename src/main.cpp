#include <glad/glad.h>
//
#include <GLFW/glfw3.h>

#include <functional>
#include <iostream>
#include <thread>

#include "SceneBuilder.hpp"

int main(int argc, char *argv[]) {
    // initialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create window
    GLFWwindow *window = glfwCreateWindow(1400, 1000, "FluidSimulation", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, RenderSystem::framebuffer_size_callback);
    glfwSetCursorPosCallback(window, RenderSystem::mouse_callback);
    glfwSetScrollCallback(window, RenderSystem::scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_CULL_FACE);

    // build scene
    auto [render_system, fluid_solver] = SceneBuilder::build_scene("pbf_cpu");
    std::thread simulation_thread([&] {
        while (!glfwWindowShouldClose(window)) {
            while (fluid_solver.is_running()) {
                fluid_solver.simulate();
            }
        }
    });

    // timing
    float delta_time = 0.0f;
    float last_time_point = 0.0f;
    // main loop
    while (!glfwWindowShouldClose(window)) {
        // calculate delta time
        float current_time_point = glfwGetTime();
        delta_time = current_time_point - last_time_point;
        last_time_point = current_time_point;
        // std::cout << 1 / deltaTime << "fps\r" << std::flush;

        // input
        render_system.process_keyboard_input(window, delta_time);

        // update
        render_system.update_particles(fluid_solver.get_partical_position());

        // render
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render_system.render();

        // swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    fluid_solver.terminate();
    simulation_thread.join();
    glfwTerminate();
    return 0;
}
