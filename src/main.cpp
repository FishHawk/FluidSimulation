#include <functional>
#include <iostream>
#include <thread>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // include glfw3.h after opengl definitions

#include "SceneBuilder.hpp"

static GLFWwindow *window;

void initialize_glfw() {
    // Initialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window = glfwCreateWindow(1400, 1000, "FluidSimulation", NULL, NULL);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    // Set callback function
    glfwSetFramebufferSizeCallback(window, render::RenderSystem::framebuffer_size_callback);
    glfwSetCursorPosCallback(window, render::RenderSystem::mouse_callback);
    glfwSetScrollCallback(window, render::RenderSystem::scroll_callback);

    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void initialize_glad() {
    // Load all opengl function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }

    // Enable opengl capabilities
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_CULL_FACE);
}

void initialize_imgui() {
    // Setup imgui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup imgui style
    ImGui::StyleColorsDark();

    // Setup platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
};

int main(int argc, char *argv[]) {
    // initialization
    initialize_glfw();
    initialize_glad();
    initialize_imgui();

    // build scene
    std::string device;
    if (argc == 1)
        device = "cpu";
    else
        device = argv[1];
    auto [render_system, fluid_system] = SceneBuilder::build_scene(device);
    std::thread simulation_thread([&] {
        while (!fluid_system.is_terminated()) {
            while (fluid_system.is_running()) {
                fluid_system.simulate();
            }
        }
    });

    // Main loop
    float delta_time = 0.0f;
    float last_time_point = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        // calculate delta time
        float current_time_point = glfwGetTime();
        delta_time = current_time_point - last_time_point;
        last_time_point = current_time_point;
        // std::cout << 1 / deltaTime << "fps\r" << std::flush;

        // Clear
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render world
        render_system.process_keyboard_input(window, delta_time);
        render_system.update_particles(fluid_system.get_particle_position());
        render_system.render();

        // Start new imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Define gui
        ImGui::Begin("Hello, world!");
        if (!fluid_system.is_running() && ImGui::Button("Start"))
            fluid_system.start();
        else if (fluid_system.is_running() && ImGui::Button("Stop"))
            fluid_system.stop();
        if (ImGui::Button("Reset"))
            ;
        ImGui::Text("This is some useful text.");
        ImGui::End();

        // Render gui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // terminate
    fluid_system.stop();
    fluid_system.terminate();
    simulation_thread.join();
    glfwTerminate();
    return 0;
}
