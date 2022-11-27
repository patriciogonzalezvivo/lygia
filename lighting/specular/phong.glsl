#include "../../math/powFast.glsl"
#include "../toShininess.glsl"

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
float specularPhong(const in vec3 L, const in vec3 N, const in vec3 V, const in float shininess) {
    vec3 R = reflect(L, N); // 2.0 * dot(N, L) * N - L;
    return SPECULAR_POW(max(0.0, dot(R, -V)), shininess);
}

float specularPhongRoughness(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) {
    return specularPhong(L, N, V, toShininess(roughness, 0.0) );
}

float specularPhongRoughness(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness, const in float fresnel) {
    return specularPhongRoughness(L, N, V, roughness );
}

float specularPhongRoughness(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness, const in float fresnel) {
    return specularPhongRoughness(L, N, V, roughness);
}

#endif