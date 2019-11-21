#include "SceneBuilder.hpp"

void add_wall(std::vector<glm::vec3> &boundary_particles,
              const glm::vec3 &min_point,
              const glm::vec3 &max_point) {
    const double diam = 2.0 * 0.025;
    const glm::vec3 diff = max_point - min_point;
    const unsigned int stepsX = static_cast<unsigned int>(diff.x / diam) + 1;
    const unsigned int stepsY = static_cast<unsigned int>(diff.y / diam) + 1;
    const unsigned int stepsZ = static_cast<unsigned int>(diff.z / diam) + 1;

    for (unsigned int x = 0; x < stepsX; x++) {
        for (unsigned int y = 0; y < stepsY; y++) {
            for (unsigned int z = 0; z < stepsZ; z++) {
                const glm::vec3 position = min_point + glm::vec3(x * diam, y * diam, z * diam);
                boundary_particles.push_back(position);
            }
        }
    }
}

std::vector<glm::vec3> SceneBuilder::init_fluid_particles() {
    std::vector<glm::vec3> fluid_particles;
    for (int z = 0; z < 15; z += 1) {
        for (int y = 0; y < 20; y += 1) {
            for (int x = 0; x < 15; x += 1) {
                fluid_particles.push_back(glm::vec3((float)x * 0.1, (float)y * 0.1, (float)z * 0.1));
            }
        }
    }
    return fluid_particles;
}

std::vector<glm::vec3> SceneBuilder::init_boundary_particles() {
    std::vector<glm::vec3> boundary_particles;

    const double container_x = 5, container_z = 5, container_y = 2.5;
    const double x1 = -container_x * 0.5;
    const double x2 = container_x * 0.5;
    const double y1 = 0.0;
    const double y2 = container_y;
    const double z1 = -container_z * 0.5;
    const double z2 = container_z * 0.5;

    add_wall(boundary_particles, glm::vec3(x1, y1, z1), glm::vec3(x2, y1, z2));
    add_wall(boundary_particles, glm::vec3(x1, y2, z1), glm::vec3(x2, y2, z2));
    add_wall(boundary_particles, glm::vec3(x1, y1, z1), glm::vec3(x1, y2, z2));
    add_wall(boundary_particles, glm::vec3(x2, y1, z1), glm::vec3(x2, y2, z2));
    add_wall(boundary_particles, glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z1));
    add_wall(boundary_particles, glm::vec3(x1, y1, z2), glm::vec3(x2, y2, z2));

    return boundary_particles;
}

void SceneBuilder::build_scene(RenderSystem &render_system, FluidSolver &fluid_solver) {
    auto fluid_particles = SceneBuilder::init_fluid_particles();
    auto boundary_particles = SceneBuilder::init_boundary_particles();
    fluid_solver.setup_model(fluid_particles, boundary_particles);
}