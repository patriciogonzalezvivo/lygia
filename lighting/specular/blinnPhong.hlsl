#include "../../math/powFast.hlsl"
#include "../toShininess.hlsl"

#ifndef SPECULAR_POW
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_POW(A,B) powFast(A,B)
#else
#define SPECULAR_POW(A,B) pow(A,B)
#endif
#endif

#ifndef FNC_SPECULAR_BLINNPHONG
#define FNC_SPECULAR_BLINNPHONG

// https://github.com/glslify/glsl-specular-blinn-phong
float specularBlinnPhong(float3 L, float3 N, float3 V, float shininess) {
    // halfVector
    float3 H = normalize(L + V);
    return SPECULAR_POW(max(0.0, dot(N, H)), shininess);
}

float specularBlinnPhongRoughnes(float3 L, float3 N, float3 V, float roughness) {
    return specularBlinnPhong(L, N, V, toShininess(roughness, 0.0) );
}

float specularBlinnPhongRoughnes(float3 L, float3 N, float3 V, float roughness, float fresnel) {
    return specularBlinnPhongRoughnes(L, N, V, roughness);
}

float specularBlinnPhongRoughnes(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness, float fresnel) {
    return specularBlinnPhongRoughnes(L, N, V, roughness);
}

#endif