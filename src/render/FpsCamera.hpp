#ifndef RENDER_FPS_CAMERA_HPP
#define RENDER_FPS_CAMERA_HPP

#include "Camera.hpp"

namespace render {

class FpsCamera : public Camera {
public:
    FpsCamera(glm::vec3 position, float yaw, float pitch, float frame_ratio)
        : Camera(position, yaw, pitch, frame_ratio) {}

    glm::mat4 view_matrix() override {
        return glm::lookAt(position_, position_ + front_, up_);
    }

    glm::mat4 projection_matrix() override {
        return glm::perspective(glm::radians(zoom_), frame_ratio_, 0.1f, 100.0f);
    }
};

} // namespace render

#endif