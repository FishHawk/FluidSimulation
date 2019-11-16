#ifndef FPS_CAMERA_HPP
#define FPS_CAMERA_HPP

#include "Camera.hpp"

class FpsCamera : public Camera {
public:
    FpsCamera(glm::vec3 position, float yaw, float pitch, float frame_ratio)
        : Camera(position, yaw, pitch, frame_ratio) {}

    glm::mat4 get_view_matrix() {
        return glm::lookAt(position_, position_ + front_, up_);
    }

    glm::mat4 get_projection_matrix() {
        return glm::perspective(glm::radians(zoom_), frame_ratio_, 0.1f, 100.0f);
    }
};

#endif