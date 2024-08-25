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
float specularPhong(ShadingData shadingData) {
    vec3 R = -reflect(shadingData.L, shadingData.N);
    return SPECULAR_POW(max(0.0, dot(R, shadingData.V)), 1.0-shadingData.linearRoughness);
}

#endif