#include "Particles.hpp"

#include <algorithm>

void Particles::reserve(std::size_t size) {
    masses.reserve(size);
    inv_masses.reserve(size);

    positions.reserve(size);
    velocities.reserve(size);
    accelerations.reserve(size);

    old_positions.reserve(size);
    last_positions.reserve(size);
    rest_positions.reserve(size);
}

void Particles::clear() {
    masses.clear();
    inv_masses.clear();

    positions.clear();
    velocities.clear();
    accelerations.clear();

    old_positions.clear();
    last_positions.clear();
    rest_positions.clear();
}

void Particles::reset() {
    positions = rest_positions;
    std::fill(velocities.begin(), velocities.end(), glm::vec3(0.0f));
    std::fill(accelerations.begin(), accelerations.end(), glm::vec3(0.0f));

    old_positions = rest_positions;
    last_positions = rest_positions;
}

void Particles::add(double mass, glm::vec3 position) {
    masses.push_back(mass);
    inv_masses.push_back((mass != 0) ? (1.0 / mass) : 0.0);

    positions.push_back(position);
    velocities.push_back(glm::vec3(0.0f));
    accelerations.push_back(glm::vec3(0.0f));

    old_positions.push_back(position);
    last_positions.push_back(position);
    rest_positions.push_back(position);
}