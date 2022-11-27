#include "../common/beckmann.glsl"

#ifndef FNC_SPECULAR_BECKMANN
#define FNC_SPECULAR_BECKMANN

float specularBeckmann(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) {
    float NoH = dot(N, normalize(L + V));
    return beckmann(NoH, roughness);
}

float specularBeckmann(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness, const in float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

float specularBeckmann(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness, const in float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

#endif