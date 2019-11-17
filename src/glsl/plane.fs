#version 330 core
out vec4 FragColor;

in vec3 pos;  
in vec3 norm; 

uniform vec3 view_pos;

vec3 lights[4] = vec3[](vec3(-5.0f, 7.0f, -5.0f), 
                        vec3(-5.0f, 7.0f,  5.0f),
                        vec3( 5.0f, 7.0f, -5.0f),
                        vec3( 5.0f, 7.0f,  5.0f));

vec3 CalcPointLight(vec3 light, vec3 normal, vec3 pos, vec3 view_dir);

void main() {
    vec3 norm = normalize(norm);
    vec3 view_dir = normalize(view_pos - pos);

    vec3 result = vec3(0.0f, 0.0f, 0.0f);
    for(int i = 0; i < 4; i++)
        result += CalcPointLight(lights[i], norm, pos, view_dir);

    FragColor = vec4(result, 1.0);
}

// calculates the color when using a point light.
vec3 CalcPointLight(vec3 light_pos, vec3 normal, vec3 pos, vec3 view_dir) {
    vec3 light_dir = normalize(light_pos - pos);
    // diffuse shading
    float diff = max(dot(normal, light_dir), 0.0);
    // specular shading
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    // attenuation
    float distance = length(light_pos - pos);
    float attenuation = 1.0 / (1.0 + 0.045 * distance + 0.0075 * (distance * distance));    
    // combine results
    vec3 ambient = 0.1 * vec3(1.0f, 1.0f, 1.0f);
    vec3 diffuse = 1.0 * diff * vec3(1.0f, 1.0f, 1.0f);
    vec3 specular = 0.5 * spec * vec3(1.0f, 1.0f, 1.0f);
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}