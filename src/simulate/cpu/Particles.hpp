#ifndef SIMULATE_CPU_PARTICLES_HPP
#define SIMULATE_CPU_PARTICLES_HPP

#include <vector>

#include <glm/glm.hpp>

namespace simulate {
namespace cpu {

struct Particles {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> velocities;
    std::vector<glm::vec3> predicted_positions;
    std::vector<glm::vec3> delta_positions;
    std::vector<glm::vec3> initial_positions;
    std::vector<float> lambdas;

    std::size_t size() const { return positions.size(); }
    void reserve(std::size_t size);
    void clear();
    void reset();
    void add(glm::vec3 position);
};

} // namespace cpu
} // namespace simulate

#endif