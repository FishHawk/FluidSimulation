#ifndef FLUID_SLOVER_HPP
#define FLUID_SLOVER_HPP

#include <glm/glm.hpp>
#include <vector>

namespace Simulation {
class FluidSolver {
private:
    bool is_running_ = true;

public:
    // is running
    bool is_running() { return is_running_; };
    void terminate() { is_running_ = false; };

    virtual void simulate() = 0;
    virtual std::vector<glm::vec3> get_partical_position() = 0;
};
}  // namespace Simulation

#endif