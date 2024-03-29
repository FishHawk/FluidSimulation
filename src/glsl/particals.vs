#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_color;
layout (location = 2) in vec3 in_norm;
layout (location = 3) in vec3 in_offset;

out vec3 pos;
out vec3 color;
out vec3 norm;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	pos = vec3(model * vec4(in_pos, 1.0));
    norm = mat3(transpose(inverse(model))) * in_norm;  
    color = in_color;
    
    gl_Position = projection * view * vec4(pos + in_offset, 1.0);
}