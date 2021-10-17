#ifndef FNC_SPECULAR_GAUSSIAN
#define FNC_SPECULAR_GAUSSIAN

// https://github.com/glslify/glsl-specular-gaussian
float specularGaussian(vec3 L, vec3 N, vec3 V, float roughness) {
    vec3 H = normalize(L + V);
    float theta = acos(dot(H, N));
    float w = theta / roughness;
    return exp(-w*w);
}

float specularGaussian(vec3 L, vec3 N, vec3 V, float roughness, float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

float specularGaussian(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

#endif