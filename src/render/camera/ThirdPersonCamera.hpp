#ifndef RENDER_THIRD_PERSON_CAMERA_HPP
#define RENDER_THIRD_PERSON_CAMERA_HPP

#include "Camera.hpp"

namespace render {

class ThirdPersonCamera : public Camera {
private:
    float back_offset_ = 10;
    float back_sensitivity_ = 1;

public:
    ThirdPersonCamera(glm::vec3 position, float yaw, float pitch, float frame_ratio, float back_offset)
        : Camera(position, yaw, pitch, frame_ratio),
          back_offset_(back_offset) {}

    glm::mat4 view_matrix() override {
        return glm::lookAt(position_ - back_offset_ * front_, position_, up_);
    }

    void slide(float offset) {
        back_offset_ -= back_sensitivity_ * offset;
        if (back_offset_ < 1) back_offset_ = 1;
    }
};

} // namespace render

#endif