#ifndef FLUID_HPP
#define FLUID_HPP

#include <glm/glm.hpp>
#include <vector>

class FluidParticle {
private:
    std::vector<double> masses_;
    std::vector<double> inv_masses_;

    std::vector<glm::vec3> position_;
    std::vector<glm::vec3> velocity_;
    std::vector<glm::vec3> acceleration_;

    std::vector<glm::vec3> old_position_;
    std::vector<glm::vec3> rest_position_;
    std::vector<glm::vec3> last_position_;

public:
    FluidParticle() = default;
    ~FluidParticle() {}

    void add(const glm::vec3 &vertex) {
        masses_.push_back(1.0);
        inv_masses_.push_back(1.0);
        position_.push_back(vertex);
        velocity_.push_back(glm::vec3(0.0f));
        acceleration_.push_back(glm::vec3(0.0f));

        old_position_.push_back(vertex);
        rest_position_.push_back(vertex);
        last_position_.push_back(vertex);
    }

    void set_mass(const unsigned int &index, const double &mass) {
        masses_[index] = mass;
        inv_masses_[index] = (mass != 0) ? (1.0 / mass) : 0.0;
    }
    void set_position(const unsigned int &index, const glm::vec3 &pos) {
        position_[index] = pos;
    }
    void set_velocity(const unsigned int &index, const glm::vec3 &vel) {
        velocity_[index] = vel;
    }
    void set_acceleration(const unsigned int &index, const glm::vec3 &acc) {
        acceleration_[index] = acc;
    }
    void set_old_position(const unsigned int &index, const glm::vec3 &pos) {
        old_position_[index] = pos;
    }
    void set_rest_position(const unsigned int &index, const glm::vec3 &pos) {
        rest_position_[index] = pos;
    }
    void set_last_position(const unsigned int &index, const glm::vec3 &pos) {
        last_position_[index] = pos;
    }


    unsigned int size() const { return position_.size(); }
    double &get_mass(const unsigned int &index) { return masses_[index]; }
    double &get_inv_mass(const unsigned int &index) { return inv_masses_[index]; }
    glm::vec3 &get_position(const unsigned int &index) { return position_[index]; }
    glm::vec3 &get_velocity(const unsigned int &index) { return velocity_[index]; }
    glm::vec3 &get_acceleration(const unsigned int &index) { return acceleration_[index]; }
    glm::vec3 &get_old_position(const unsigned int &index) { return old_position_[index]; }
    glm::vec3 &get_rest_position(const unsigned int &index) { return rest_position_[index]; }
    glm::vec3 &get_last_position(const unsigned int &index) { return last_position_[index]; }

    std::vector<double> &get_mass_vector() { return masses_; }
    std::vector<double> &get_inv_mass_vector() { return inv_masses_; }
    std::vector<glm::vec3> &get_position_vector() { return position_; }
    std::vector<glm::vec3> &get_velocity_vector() { return velocity_; }
    std::vector<glm::vec3> &get_acceleration_vector() { return acceleration_; }
    std::vector<glm::vec3> &get_old_position_vector() { return old_position_; }
    std::vector<glm::vec3> &get_rest_position_vector() { return rest_position_; }
    std::vector<glm::vec3> &get_last_position_vector() { return last_position_; }
};
#endif