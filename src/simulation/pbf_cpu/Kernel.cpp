#include "Kernel.hpp"

using namespace Simulation::PbfCpu;

double Kernel::poly6_kernel(const glm::vec3& r, double h) {
    double ret = 0.0;
    double rl = glm::length(r);
    double q = rl / h;
    double h3 = h * h * h;

    if (q > 1) return 0;

    if (q <= 0.5) {
        double q2 = q * q;
        double q3 = q2 * q;
        ret = 8.0 / (M_PI * h3) * (6.0 * q3 - 6.0 * q2 + 1.0);
    } else {
        ret = 16.0 / (M_PI * h3) * pow(1 - q, 3.0);
    }
    return ret;
}

glm::vec3 Kernel::poly6_kernal_grade(const glm::vec3& r, double h) {
    glm::vec3 ret(0.0f);
    double rl = glm::length(r);
    double q = rl / h;
    double h3 = h * h * h;

    if (q > 1) return ret;

    if (rl > 1.0e-6) {
        const glm::vec3 gradq = static_cast<float>(1.0 / (rl * h)) * r;
        if (q <= 0.5) {
            ret = static_cast<float>(48.0 / (M_PI * h3) * q * (3.0 * q - 2.0)) * gradq;
        } else {
            double factor = 1.0 - q;
            ret = static_cast<float>(48.0 / (M_PI * h3) * (-factor * factor)) * gradq;
        }
    }
    return ret;
}