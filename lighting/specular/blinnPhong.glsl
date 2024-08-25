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
float specularBlinnPhong(ShadingData shadingData) {
    return SPECULAR_POW(shadingData.NoH, 1.0-shadingData.linearRoughness);
}

#endif