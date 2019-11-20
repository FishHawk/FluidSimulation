#include "FluidSolver.hpp"

void FluidSolver::simulation() {
}

std::vector<glm::vec3> FluidSolver::get_partical_position() {
    std::vector<glm::vec3> positions;
    for (int z = 0; z < 15; z += 1) {
        for (int y = 0; y < 20; y += 1) {
            for (int x = 0; x < 15; x += 1) {
                positions.push_back(glm::vec3((float)x * 0.1, (float)y * 0.1, (float)z * 0.1));
            }
        }
    }
    return positions;
}