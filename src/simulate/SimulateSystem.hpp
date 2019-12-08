#ifndef SIMULATE_SIMULATE_SYSTEM_HPP
#define SIMULATE_SIMULATE_SYSTEM_HPP

#include <vector>

#include <glm/glm.hpp>

namespace simulate {

class SimulateSystem {
private:
    bool is_running_ = false;
    bool is_terminated_ = false;

public:
    bool is_running() { return is_running_; };
    void start() { is_running_ = true; };
    void stop() { is_running_ = false; };

    bool is_terminated() { return is_terminated_; };
    void terminate() { is_terminated_ = true; };

    virtual void simulate() = 0;
    // virtual void reset() = 0;

    virtual std::vector<glm::vec3> get_particle_position() = 0;
};

} // namespace simulate

#endif