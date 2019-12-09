#include <functional>
#include <iostream>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // include glfw3.h after opengl definitions

#include "Input.hpp"
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
    glfwSetFramebufferSizeCallback(window, Input::framebuffer_size_callback);
    glfwSetCursorPosCallback(window, Input::mouse_callback);
    glfwSetScrollCallback(window, Input::scroll_callback);
}

void initialize_glad() {
    // Load all opengl function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }
}

void initialize_imgui() {
    // setup imgui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // setup imgui io
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;

    // setup imgui style
    ImGui::StyleColorsDark();

    // setup platform/renderer bindings
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
    auto [render_system, simulate_system] = SceneBuilder::build_scene(device);

    // Main loop
    float delta_time = 0.0f;
    float last_time_point = 0.0f;
    bool render_axis = false, render_container = false;
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
        Input::process_keyboard_input(window, delta_time);
        render_system.update_particles(simulate_system.get_particle_position());
        render_system.render();

        // Start new imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Define gui
        ImGui::Begin("Console");

        ImGui::Text("Simulate System");
        if (!simulate_system.is_running() && ImGui::Button("Start"))
            simulate_system.start();
        else if (simulate_system.is_running() && ImGui::Button("Stop"))
            simulate_system.stop();
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            simulate_system.stop();
            simulate_system.reset();
        }
        ImGui::Separator();

        ImGui::Text("Render System");
        ImGui::Checkbox("Render Axis", render_system.get_axis_switch());
        ImGui::Checkbox("Render Container", render_system.get_container_switch());

        ImGui::End();

        // Render gui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // terminate
    simulate_system.terminate();
    glfwTerminate();
    return 0;
}
