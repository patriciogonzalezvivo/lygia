#include "../common/beckmann.hlsl"

#ifndef FNC_SPECULAR_BECKMANN
#define FNC_SPECULAR_BECKMANN

float specularBeckmann(float3 L, float3 N, float3 V, float roughness) {
    float NoH = dot(N, normalize(L + V));
    return beckmann(NoH, roughness);
}

float specularBeckmann(float3 L, float3 N, float3 V, float roughness, float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

float specularBeckmann(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularBeckmann(L, N, V, roughness);
}

#endif