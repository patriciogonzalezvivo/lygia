#include "../../math/powFast.glsl"
#include "../toShininess.glsl"

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
float specularBlinnPhong(const in vec3 L, const in vec3 N, const in vec3 V, float shininess) {
    // halfVector
    vec3 H = normalize(L + V);
    return SPECULAR_POW(max(0.0, dot(N, H)), shininess);
}

float specularBlinnPhongRoughnes(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) {
    return specularBlinnPhong(L, N, V, toShininess(roughness, 0.0) );
}

float specularBlinnPhongRoughnes(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness, const in float fresnel) {
    return specularBlinnPhongRoughnes(L, N, V, roughness);
}

float specularBlinnPhongRoughnes(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness, const in float fresnel) {
    return specularBlinnPhongRoughnes(L, N, V, roughness);
}

#endif