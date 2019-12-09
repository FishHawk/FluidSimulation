#ifndef SIMULATE_SIMULATE_SYSTEM_HPP
#define SIMULATE_SIMULATE_SYSTEM_HPP

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

namespace simulate {

class SimulateSystem {
private:
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_terminated_{false};

protected:
    std::mutex m;
    std::thread simulate_thread;

    // container
    glm::vec3 container_start_ = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 container_end_ = glm::vec3(1.0f, 1.0f, 1.0f);

public:
    SimulateSystem() {
        simulate_thread = std::thread([&] {
            while (!is_terminated_) {
                while (is_running_) {
                    std::lock_guard<std::mutex> lock(m);
                    simulate();
                }
            }
        });
    }

    // run
    bool is_running() { return is_running_; };
    void start() { is_running_ = true; };
    void stop() { is_running_ = false; };

    // terminate
    bool is_terminated() { return is_terminated_; };
    void terminate() {
        is_running_ = false;
        is_terminated_ = true;
        simulate_thread.join();
    };

    // simulate
    virtual void simulate() = 0;
    virtual void reset() = 0;

    // container
    void set_container(glm::vec3 container_position, glm::vec3 container_size) {
        container_start_ = container_position;
        container_end_ = container_position + container_size;
    }

    // fluid particles
    virtual void set_particles_position(const std::vector<glm::vec3> &particles_initial_positions) = 0;
    virtual void set_particles_radius(double radius) = 0;

    virtual std::vector<glm::vec3> get_particle_position() = 0;
};

} // namespace simulate

#endif