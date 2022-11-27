#include "common/schlick.glsl"
#include "../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <float|vec3> fresnel(const <float|vec3> f0, <float> NoV)
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const in vec3 f0, const in float NoV) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    return schlick(f0, 1.0, NoV);
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return schlick(f0, f90, NoV);
#endif
}

float fresnel(const in float f0, const in float NoV) {
    return schlick(f0, 1.0, NoV);
}
#endif