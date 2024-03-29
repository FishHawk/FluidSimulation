#include "Particles.hpp"

#include <algorithm>

using namespace simulate::cpu;

void Particles::reserve(std::size_t size) {
    positions.reserve(size);
    velocities.reserve(size);

    delta_positions.reserve(size);
    predicted_positions.reserve(size);
    initial_positions.reserve(size);

    lambdas.reserve(size);
}

void Particles::clear() {
    positions.clear();
    velocities.clear();

    delta_positions.clear();
    predicted_positions.clear();
    initial_positions.clear();

    lambdas.clear();
}

void Particles::reset() {
    positions = initial_positions;
    predicted_positions = initial_positions;
    std::fill(velocities.begin(), velocities.end(), glm::vec3(0.0f));
}

void Particles::add(glm::vec3 position) {
    positions.push_back(position);
    velocities.push_back(glm::vec3(0.0f));

    delta_positions.push_back(glm::vec3(0.0f));
    predicted_positions.push_back(position);
    initial_positions.push_back(position);

    lambdas.push_back(0.0f);
}