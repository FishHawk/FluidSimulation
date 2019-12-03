#ifndef RENDER_CAMERA_HPP
#define RENDER_CAMERA_HPP

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace render {

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
          yaw_(yaw),
          pitch_(pitch),
          frame_ratio_(frame_ratio) {
        update_direction_vector();
    }

    const glm::vec3 position();
    const glm::vec3 front();

    virtual glm::mat4 view_matrix() = 0;
    virtual glm::mat4 projection_matrix() = 0;

    // update camera
    void move(MovementDirection direction, float delta_time);
    void rotate(float xoffset, float yoffset, GLboolean constrain_pitch = true);
    void zoom(float yoffset);
    void change_frame_ratio(float frame_ratio);

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

    void update_direction_vector();
};

} // namespace render

#endif