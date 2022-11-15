#include "common/schlick.glsl"
#include "../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <vec3> fresnel(const <vec3> f0, <float> NoV)
    - <vec3> fresnel(<vec3> V <vec3> N, <float> R0)
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const vec3 f0, float NoV) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    return schlick(f0, 1.0, NoV);
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return schlick(f0, f90, NoV);
#endif
}

// float fresnelf(vec3 V, vec3 N, float R0) {
//     float cosAngle = 1.0-max(dot(V, N), 0.0);
//     float result = cosAngle * cosAngle;
//     result = result * result;
//     result = result * cosAngle;
//     result = clamp(result * (1.0 - R0) + R0, 0.0, 1.0);
//     return result;
// }

#endif