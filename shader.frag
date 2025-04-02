#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 dir;
layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform UniformBufferObject {
    vec2 resolution;
    float len;
    vec3 origin;
    mat4 view;
} ubo;

float kernal(vec3 ver) {
    vec3 a = ver;
    float b, c, d;
    for(int i = 0; i < 5; i++) {
        b = length(a);
        c = atan(a.y, a.x) * 8.0;
        d = acos(a.z / b) * 8.0;
        b = pow(b, 8.0);
        a = vec3(b * sin(d) * cos(c), b * sin(d) * sin(c), b * cos(d)) + ver;
        if(b > 6.0) break;
    }
    return 4.0 - dot(a, a);
}

void main() {
    vec3 rayDir = normalize(dir);
    float t = 0.0;
    float step = 0.01 * ubo.len;  // 原始step=0.002;加大step可以有效降低计算量
    float maxDist = 2.0 * ubo.len;
    int sign = 0;
    vec3 ver;

    for(int k = 0; k < 1000; k++) {
        ver = ubo.origin + rayDir * t;
        float v = kernal(ver);
        if(v > 0.0) {
            vec3 n;
            float eps = t * 0.00025;
            n.x = kernal(ver - vec3(eps, 0.0, 0.0)) - kernal(ver + vec3(eps, 0.0, 0.0));
            n.y = kernal(ver - vec3(0.0, eps, 0.0)) - kernal(ver + vec3(0.0, eps, 0.0));
            n.z = kernal(ver - vec3(0.0, 0.0, eps)) - kernal(ver + vec3(0.0, 0.0, eps));
            n = normalize(n);

            float r = length(ver);
            vec3 color = vec3(
                sin(r * 10.0) * 0.5 + 0.5,
                sin(r * 10.0 + 2.05) * 0.5 + 0.5,
                sin(r * 10.0 - 2.05) * 0.5 + 0.5
            );
            float diff = max(0.0, dot(n, -rayDir));
            fragColor = vec4(color * (diff * 0.45 + 0.3), 1.0);
            return;
        }
        t += step;
        if(t > maxDist) break;
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
