#ifndef UI_HPP
#define UI_HPP

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "render/RenderSystem.hpp"
#include "simulate/SimulateSystem.hpp"

class Ui {
private:
    static render::RenderSystem *render_system_;
    static simulate::SimulateSystem *simulate_system_;

public:
    static void link_to_systems(render::RenderSystem &render_system, simulate::SimulateSystem &simulate_system) {
        render_system_ = &render_system;
        simulate_system_ = &simulate_system;
    }

    static void init(GLFWwindow *window) {
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
    }

    static void render() {
        static bool render_axis = false, render_container = false;
        static std::vector<float> relative_speed_history;
        relative_speed_history.resize(60 * 3);

        // Start new imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Define gui
        ImGui::Begin("Console");

        ImGui::Text("Info");

        relative_speed_history.erase(relative_speed_history.begin());
        relative_speed_history.push_back(simulate_system_->get_relative_speed());

        float relative_speed_average = 0;
        for (auto &s : relative_speed_history)
            relative_speed_average += s;
        relative_speed_average /= relative_speed_history.size();

        ImGui::Text("relative speed = %f", relative_speed_average);
        ImGui::PlotLines("", relative_speed_history.data(), relative_speed_history.size(),
                         0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 30));
        ImGui::Text("real time / simulate time");
        ImGui::Separator();

        ImGui::Text("Simulate System");
        if (!simulate_system_->is_running() && ImGui::Button("Start"))
            simulate_system_->start();
        else if (simulate_system_->is_running() && ImGui::Button("Stop"))
            simulate_system_->stop();
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            simulate_system_->stop();
            simulate_system_->reset();
        }
        ImGui::Separator();

        ImGui::Text("Render System");
        ImGui::Checkbox("Render Axis", render_system_->get_axis_switch());
        ImGui::Checkbox("Render Container", render_system_->get_container_switch());

        ImGui::End();

        // Render gui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
};

#endif