#ifndef FNC_SPECULAR_GAUSSIAN
#define FNC_SPECULAR_GAUSSIAN

// https://github.com/glslify/glsl-specular-gaussian
float specularGaussian(float3 L, float3 N, float3 V, float roughness) {
    float3 H = normalize(L + V);
    float theta = acos(dot(H, N));
    float w = theta / roughness;
    return exp(-w*w);
}

float specularGaussian(float3 L, float3 N, float3 V, float roughness, float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

float specularGaussian(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularGaussian(L, N, V, roughness);
}

#endif