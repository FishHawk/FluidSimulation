#ifndef SIMULATE_CPU_PARTICLES_HPP
#define SIMULATE_CPU_PARTICLES_HPP

#include <glm/glm.hpp>
#include <vector>

namespace simulate {
namespace cpu {

struct Particles {
    std::vector<double> masses;
    std::vector<double> inv_masses;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> velocities;
    std::vector<glm::vec3> accelerations;

    std::vector<glm::vec3> old_positions;
    std::vector<glm::vec3> last_positions;
    std::vector<glm::vec3> rest_positions;

    std::size_t size() const { return positions.size(); }
    void reserve(std::size_t size);
    void clear();
    void reset();
    void add(double mass, glm::vec3 position);
};

} // namespace cpu
} // namespace simulate

#endif