#include "SceneBuilder.hpp"

#include "simulate/cpu/SimulateSystem.hpp"
#include "simulate/cuda/SimulateSystem.hpp"

void fill(std::vector<glm::vec3> &boundary_particles,
          double particle_radius,
          const glm::vec3 &min_point,
          const glm::vec3 &max_point) {
    const double diam = 2.0 * particle_radius;
    const glm::vec3 diff = max_point - min_point;
    const unsigned int stepsX = static_cast<unsigned int>(diff.x / diam) + 1;
    const unsigned int stepsY = static_cast<unsigned int>(diff.y / diam) + 1;
    const unsigned int stepsZ = static_cast<unsigned int>(diff.z / diam) + 1;

    for (int x = 0; x < stepsX; x++) {
        for (int y = 0; y < stepsY; y++) {
            for (int z = 0; z < stepsZ; z++) {
                const glm::vec3 position = min_point + glm::vec3(x * diam, y * diam, z * diam);
                boundary_particles.push_back(position);
            }
        }
    }
}

std::vector<glm::vec3> SceneBuilder::init_fluid_particles(double particle_radius) {
    std::vector<glm::vec3> fluid_particles;

    const double p = 2;
    const double fluid_x = p * 0.5, fluid_z = p * 0.5, fluid_y = p * 0.5;
    const double x1 = -fluid_x * 0.5;
    const double x2 = fluid_x * 0.5;
    const double y1 = 2.0 - fluid_y;
    const double y2 = 2.0;
    const double z1 = -fluid_z * 0.5;
    const double z2 = fluid_z * 0.5;

    fill(fluid_particles, particle_radius, glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z2));

    return fluid_particles;
}

std::vector<glm::vec3> SceneBuilder::init_boundary_particles(double particle_radius) {
    std::vector<glm::vec3> boundary_particles;

    const double container_x = 2, container_z = 2, container_y = 4;
    const double x1 = -container_x * 0.5;
    const double x2 = container_x * 0.5;
    const double y1 = 0.0;
    const double y2 = container_y;
    const double z1 = -container_z * 0.5;
    const double z2 = container_z * 0.5;

    fill(boundary_particles, particle_radius, glm::vec3(x1, y1, z1), glm::vec3(x2, y1, z2)); // y-
    // fill(boundary_particles, particle_radius, glm::vec3(x1, y2, z1), glm::vec3(x2, y2, z2)); // y+
    fill(boundary_particles, particle_radius, glm::vec3(x1, y1, z1), glm::vec3(x1, y2, z2)); // x-
    fill(boundary_particles, particle_radius, glm::vec3(x2, y1, z1), glm::vec3(x2, y2, z2)); // x+
    fill(boundary_particles, particle_radius, glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z1)); // z-
    fill(boundary_particles, particle_radius, glm::vec3(x1, y1, z2), glm::vec3(x2, y2, z2)); // z+

    return boundary_particles;
}

simulate::SimulateSystem &SceneBuilder::get_simulate_system(std::string device) {
    if (device == "cpu")
        return simulate::cpu::SimulateSystem::get_instance();
    else if (device == "cuda")
        return simulate::cuda::SimulateSystem::get_instance();
}

std::pair<render::RenderSystem &, simulate::SimulateSystem &> SceneBuilder::build_scene(std::string device) {
    auto &render_system = render::RenderSystem::get_instance();
    auto &simulate_system = get_simulate_system(device);

    glm::vec3 container_size(2.0f, 4.0f, 2.0f);
    glm::vec3 container_position(-container_size.x * 0.5f, 0.0f, -container_size.z * 0.5f);
    render_system.set_container(container_position, container_size);
    simulate_system.set_container(container_position, container_size);

    double particle_radius = 0.025;
    render_system.set_particle_radius(particle_radius);
    simulate_system.set_particle_radius(particle_radius);

    auto fluid_particles = init_fluid_particles(particle_radius);
    simulate_system.set_particle_position(fluid_particles);

    simulate_system.apply();
    return {render_system, simulate_system};
}