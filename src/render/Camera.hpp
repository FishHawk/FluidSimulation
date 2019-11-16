#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

class Camera {
public:
    enum class MovementDirection {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT
    };

    Camera(glm::vec3 position, float yaw, float pitch, float frame_ratio)
        : position_(position),
          yaw_(yaw_),
          pitch_(pitch),
          frame_ratio_(frame_ratio) {
        update_direction_vector();
    }

    virtual glm::mat4 get_view_matrix() = 0;
    virtual glm::mat4 get_projection_matrix() = 0;

    void move(MovementDirection direction, float delta_time) {
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

    void rotate(float xoffset, float yoffset, GLboolean constrain_pitch = true) {
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

    void zoom(float yoffset) {
        if (zoom_ >= 1.0f && zoom_ <= 45.0f)
            zoom_ -= yoffset;
        if (zoom_ <= 1.0f)
            zoom_ = 1.0f;
        if (zoom_ >= 45.0f)
            zoom_ = 45.0f;
    }

    void change_frame_ratio(float frame_ratio) {
        frame_ratio_ = frame_ratio;
    }

protected:
    // config
    const glm::vec3 world_up_ = glm::vec3(0.0f, 1.0f, 0.0f);
    float move_speed_ = 2.5f;
    float rotate_sensitivity_ = 0.1f;

    // camera positon
    glm::vec3 position_;

    // camera direction
    float yaw_;
    float pitch_;

    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;

    // camera projection
    float zoom_ = 45.0f;
    float frame_ratio_;

    void update_direction_vector() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front.y = sin(glm::radians(pitch_));
        front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(front);
        right_ = glm::normalize(glm::cross(front_, world_up_));
        up_ = glm::normalize(glm::cross(right_, front_));
    }
};

#endif