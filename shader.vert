#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;

layout(binding = 0) uniform UniformBufferObject {
    vec2 resolution;
    float len;
    vec3 origin;
    mat4 view;
} ubo;

layout(location = 0) out vec3 dir;

void main() {
    gl_Position = vec4(position, 1.0);
    float aspect = ubo.resolution.x / ubo.resolution.y;
    vec3 forward = normalize(-ubo.origin);  // 从 origin 指向原点
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    dir = normalize(forward + right * position.x * aspect + up * position.y);
}
