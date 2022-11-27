#ifndef FNC_SPECULAR_GAUSSIAN
#define FNC_SPECULAR_GAUSSIAN

// https://github.com/glslify/glsl-specular-gaussian
float specularGaussian(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) {
    vec3 H = normalize(L + V);
    float theta = acos(dot(H, N));
    float w = theta / roughness;
    return exp(-w*w);
}

float specularGaussian(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness, const in float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

float specularGaussian(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness, const in float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

#endif