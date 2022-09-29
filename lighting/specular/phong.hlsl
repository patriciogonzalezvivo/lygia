#include "../../math/powFast.hlsl"
#include "../toShininess.hlsl"

#ifndef SPECULAR_POW
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_POW(A,B) powFast(A,B)
#else
#define SPECULAR_POW(A,B) pow(A,B)
#endif
#endif

#ifndef FNC_SPECULAR_PHONG
#define FNC_SPECULAR_PHONG 

// https://github.com/glslify/glsl-specular-phong
float specularPhong(float3 L, float3 N, float3 V, float shininess) {
    float3 R = reflect(L, N); // 2.0 * dot(N, L) * N - L;
    return SPECULAR_POW(max(0.0, dot(R, -V)), shininess);
}

float specularPhongRoughness(float3 L, float3 N, float3 V, float roughness) {
    return specularPhong(L, N, V, toShininess(roughness, 0.0) );
}

float specularPhongRoughness(float3 L, float3 N, float3 V, float roughness, float fresnel) {
    return specularPhongRoughness(L, N, V, roughness );
}

float specularPhongRoughness(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularPhongRoughness(L, N, V, roughness);
}

#endif