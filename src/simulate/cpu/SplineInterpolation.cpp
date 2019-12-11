#include "SplineInterpolation.hpp"

using namespace simulate::cpu;

double SplineInterpolation::poly6_kernel(const glm::vec3 &r, double h) {
    auto l = glm::length(r);
    // static double test = 315.0 / (64 * M_PI * powf(h, 9));

    if (l < h && l > 1.0e-6) {
        return 315.0 * powf((h * h - l * l), 3) / (64 * M_PI * powf(h, 9));
        // return test * powf((h * h - l * l), 3);

        // double q = glm::length(r) / h;
        // double h3 = h * h * h;
        // if (q <= 0.5) {
        //     double q2 = q * q;
        //     double q3 = q2 * q;
        //     return 8.0 / (M_PI * h3) * (6.0 * q3 - 6.0 * q2 + 1.0);
        // } else {
        //     return 16.0 / (M_PI * h3) * pow(1 - q, 3.0);
        // }
    }
    return 0.0;
}

glm::vec3 SplineInterpolation::poly6_kernal_grade(const glm::vec3 &r, double h) {
    double rl = glm::length(r);

    // static double test = -45.0 / (M_PI * powf(h, 6));
    if (rl < h && rl > 1.0e-6) {
        return float(-45.0 / (M_PI * powf(h, 6)) * (h - rl) * (h - rl) / rl) * r;
        // return float(test * (h - rl) * (h - rl) / rl) * r;

        // double q = rl / h;
        // const glm::vec3 gradq = static_cast<float>(1.0 / (rl * h)) * r;
        // double h3 = h * h * h;
        // if (q <= 0.5) {
        //     return static_cast<float>(48.0 / (M_PI * h3) * q * (3.0 * q - 2.0)) * gradq;
        // } else {
        //     double factor = 1.0 - q;
        //     return static_cast<float>(48.0 / (M_PI * h3) * (-factor * factor)) * gradq;
        // }
    }
    return glm::vec3(0.0f);
}