#ifndef SIMULATE_CPU_KERNAL_HPP
#define SIMULATE_CPU_KERNAL_HPP

#include <glm/glm.hpp>

namespace simulate {
namespace cpu {

class Kernel {
public:
    static double poly6_kernel(const glm::vec3 &r, double h);
    static glm::vec3 poly6_kernal_grade(const glm::vec3 &r, double h);
};

} // namespace cpu
} // namespace simulate

#endif