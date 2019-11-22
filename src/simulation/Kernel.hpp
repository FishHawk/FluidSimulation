#ifndef KERNAL_HPP
#define KERNAL_HPP

#include <glm/glm.hpp>

class Kernel {
public:
    static double poly6_kernel(const glm::vec3 &r, double h);
    static glm::vec3 poly6_kernal_grade(const glm::vec3 &r, double h);
};

#endif