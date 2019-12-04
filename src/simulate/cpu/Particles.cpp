#include "Particles.hpp"

#include <algorithm>

using namespace simulate::cpu;

void Particles::reserve(std::size_t size) {
    masses.reserve(size);

    positions.reserve(size);
    velocities.reserve(size);

    predicted_positions.reserve(size);
    initial_positions.reserve(size);
}

void Particles::clear() {
    masses.clear();

    positions.clear();
    velocities.clear();

    predicted_positions.clear();
    initial_positions.clear();
}

void Particles::reset() {
    positions = initial_positions;
    predicted_positions = initial_positions;
    std::fill(velocities.begin(), velocities.end(), glm::vec3(0.0f));
}

void Particles::add(double mass, glm::vec3 position) {
    masses.push_back(mass);

    positions.push_back(position);
    velocities.push_back(glm::vec3(0.0f));

    predicted_positions.push_back(position);
    initial_positions.push_back(position);
}