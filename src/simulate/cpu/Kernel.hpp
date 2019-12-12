#ifndef SIMULATE_CPU_KERNEL_HPP
#define SIMULATE_CPU_KERNEL_HPP

#include <glm/glm.hpp>

namespace simulate {
namespace cpu {

class Kernel {
private:
    static float poly6_coff;
    static float spiky_gradient_coff;

public:
    static float poly6(const glm::vec3 &r, float h) {
        auto rl = glm::length(r);

        if (rl < h && rl > 1.0e-6) {
            return 315.0 * powf((h * h - rl * rl), 3) / (64.0f * M_PI * powf(h, 9));
        }
        return 0.0f;
    }

    static glm::vec3 spiky_gradient(const glm::vec3 &r, float h) {
        float rl = glm::length(r);

        if (rl < h && rl > 1.0e-6) {
            return -45.0f * powf((h - rl), 2) / (float(M_PI) * powf(h, 6) * rl) * r;
        }
        return glm::vec3(0.0f);
    }
};

} // namespace cpu
} // namespace simulate

#endif