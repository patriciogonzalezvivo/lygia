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
float specularBlinnPhong(const in vec3 NoH, float shininess) {
    return SPECULAR_POW(max(0.0, NoH), shininess);
}

float specularBlinnPhong(ShadingData, shadingData) {
    return specularBlinnPhong(shadingData.L, shadingData.NoH, 1.0-shadingData.roughness);
}

#endif