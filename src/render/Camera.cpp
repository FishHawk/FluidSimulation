#include "Camera.hpp"

void Camera::update_direction_vector() {
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front_ = glm::normalize(front);
    right_ = glm::normalize(glm::cross(front_, world_up_));
    up_ = glm::normalize(glm::cross(right_, front_));
}

const glm::vec3 Camera::position() {
    return position_;
}
const glm::vec3 Camera::front() {
    return front_;
}

void Camera::move(MovementDirection direction, float delta_time) {
    float velocity = move_speed_ * delta_time;
    if (direction == MovementDirection::FORWARD)
        position_ += front_ * velocity;
    else if (direction == MovementDirection::BACKWARD)
        position_ -= front_ * velocity;
    else if (direction == MovementDirection::LEFT)
        position_ -= right_ * velocity;
    else if (direction == MovementDirection::RIGHT)
        position_ += right_ * velocity;
}

void Camera::rotate(float xoffset, float yoffset, GLboolean constrain_pitch) {
    xoffset *= rotate_sensitivity_;
    yoffset *= rotate_sensitivity_;

    yaw_ += xoffset;
    pitch_ += yoffset;

    if (yaw_ > 180)
        yaw_ -= 360;
    else if (yaw_ < -180)
        yaw_ += 360;

    if (constrain_pitch) {
        if (pitch_ > 89.0f)
            pitch_ = 89.0f;
        else if (pitch_ < -89.0f)
            pitch_ = -89.0f;
    }

    update_direction_vector();
}

void Camera::zoom(float yoffset) {
    if (zoom_ >= 1.0f && zoom_ <= 45.0f)
        zoom_ -= yoffset;
    if (zoom_ <= 1.0f)
        zoom_ = 1.0f;
    if (zoom_ >= 45.0f)
        zoom_ = 45.0f;
}

void Camera::change_frame_ratio(float frame_ratio) {
    frame_ratio_ = frame_ratio;
}