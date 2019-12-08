#ifndef RENDER_FIRST_PERSON_CAMERA_HPP
#define RENDER_FIRST_PERSON_CAMERA_HPP

#include "Camera.hpp"

namespace render {

class FirstPersonCamera : public Camera {
public:
    FirstPersonCamera(glm::vec3 position, float yaw, float pitch, float frame_ratio)
        : Camera(position, yaw, pitch, frame_ratio) {}

    glm::mat4 view_matrix() override {
        return glm::lookAt(position_, position_ + front_, up_);
    }
};

} // namespace render

#endif