#ifndef SIMULATE_SIMULATE_SYSTEM_HPP
#define SIMULATE_SIMULATE_SYSTEM_HPP

#include <glm/glm.hpp>
#include <vector>

namespace simulate {

class SimulateSystem {
private:
    bool is_running_ = true;

public:
    // is running
    bool is_running() { return is_running_; };
    void terminate() { is_running_ = false; };

    virtual void simulate() = 0;
    virtual std::vector<glm::vec3> get_partical_position() = 0;
};

} // namespace simulate

#endif