#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;

out vec3 pos;
out vec3 norm;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	pos = vec3(model * vec4(in_pos, 1.0));
    norm = mat3(transpose(inverse(model))) * in_norm;  
    
    gl_Position = projection * view * vec4(pos, 1.0);
}