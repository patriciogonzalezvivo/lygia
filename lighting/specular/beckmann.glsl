#include "../common/beckmann.glsl"

#ifndef FNC_SPECULAR_BECKMANN
#define FNC_SPECULAR_BECKMANN

float specularBeckmann(vec3 L, vec3 N, vec3 V, float roughness) {
    float NoH = dot(N, normalize(L + V));
    return beckmann(NoH, roughness);
}

float specularBeckmann(vec3 L, vec3 N, vec3 V, float roughness, float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

float specularBeckmann(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

#endif