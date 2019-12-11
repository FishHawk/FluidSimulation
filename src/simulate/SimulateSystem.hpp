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
    // thread
    std::mutex m;
    std::thread simulate_thread;

    // timer
    std::atomic<float> time_step_{0.016f};
    std::atomic<float> relative_speed_{0.0f};

    // container
    glm::vec3 container_start_{0.0f, 0.0f, 0.0f};
    glm::vec3 container_end_{1.0f, 1.0f, 1.0f};

    // radius
    float particle_radius_{0.025};
    float sph_radius_{4 * particle_radius_};

public:
    SimulateSystem() {
        simulate_thread = std::thread([&] {
            while (!is_terminated_) {
                while (is_running_) {
                    std::lock_guard<std::mutex> lock(m);

                    auto time_start = std::chrono::system_clock::now();
                    simulate();
                    auto time_end = std::chrono::system_clock::now();
                    auto duration = std::chrono::duration<float>(time_end - time_start).count();
                    relative_speed_ = time_step_ / duration;

                    while (duration < time_step_) {
                        time_end = std::chrono::system_clock::now();
                        duration = std::chrono::duration<float>(time_end - time_start).count();
                    }
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

    // timer
    float get_relative_speed() const {
        return relative_speed_;
    }

    // container
    void set_container(glm::vec3 container_position, glm::vec3 container_size) {
        container_start_ = container_position;
        container_end_ = container_position + container_size;
    }

    // fluid particles
    virtual void set_particle_position(const std::vector<glm::vec3> &particles_initial_positions) = 0;
    void set_particle_radius(float radius) {
        particle_radius_ = radius;
        sph_radius_ = 4 * radius;
    }

    // ! for temp use
    virtual void apply(){};
    virtual std::vector<glm::vec3> get_particle_position() = 0;
};

} // namespace simulate

#endif