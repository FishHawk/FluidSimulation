#ifndef SIMULATE_SIMULATE_SYSTEM_HPP
#define SIMULATE_SIMULATE_SYSTEM_HPP

#include <mutex>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

namespace simulate {

class SimulateSystem {
private:
    bool is_running_ = false;
    bool is_terminated_ = false;

protected:
    std::mutex m;
    std::thread simulate_thread;

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

    bool is_running() { return is_running_; };
    void start() { is_running_ = true; };
    void stop() { is_running_ = false; };

    bool is_terminated() { return is_terminated_; };
    void terminate() {
        is_running_ = false;
        is_terminated_ = true;
        simulate_thread.join();
    };

    virtual void simulate() = 0;
    virtual void reset() = 0;

    virtual std::vector<glm::vec3> get_particle_position() = 0;
};

} // namespace simulate

#endif