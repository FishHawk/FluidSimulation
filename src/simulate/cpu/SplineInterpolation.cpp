#include "SplineInterpolation.hpp"

using namespace simulate::cpu;

double SplineInterpolation::poly6_kernel(const glm::vec3 &r, double h) {
    double q = glm::length(r) / h;
    if (q > 1)
        return 0.0;

    double h3 = h * h * h;
    if (q <= 0.5) {
        double q2 = q * q;
        double q3 = q2 * q;
        return 8.0 / (M_PI * h3) * (6.0 * q3 - 6.0 * q2 + 1.0);
    } else {
        return 16.0 / (M_PI * h3) * pow(1 - q, 3.0);
    }
}

glm::vec3 SplineInterpolation::poly6_kernal_grade(const glm::vec3 &r, double h) {
    double rl = glm::length(r);
    double q = rl / h;

    if (q < 1 && rl > 1.0e-6) {
        const glm::vec3 gradq = static_cast<float>(1.0 / (rl * h)) * r;
        double h3 = h * h * h;
        if (q <= 0.5) {
            return static_cast<float>(48.0 / (M_PI * h3) * q * (3.0 * q - 2.0)) * gradq;
        } else {
            double factor = 1.0 - q;
            return static_cast<float>(48.0 / (M_PI * h3) * (-factor * factor)) * gradq;
        }
    }
    return glm::vec3(0.0f);
}