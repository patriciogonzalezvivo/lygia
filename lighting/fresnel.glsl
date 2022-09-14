#include "common/schlick.glsl"
#include "envMap.glsl"
#include "fakeCube.glsl"
#include "sphericalHarmonics.glsl"
#include "../color/tonemap.glsl"
#include "../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <vec3> fresnel(const <vec3> f0, <float> LoH)
    - <vec3> fresnel(<vec3> _R, <vec3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const vec3 f0, float LoH) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    return schlick(f0, 1.0, LoH);
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return schlick(f0, f90, LoH);
#endif
}

vec3 fresnel(vec3 _R, vec3 _f0, float _NoV) {
    vec3 frsnl = fresnel(_f0, _NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(SCENE_SH_ARRAY)
    reflectColor = tonemap( sphericalHarmonics(_R) );
    #else
    reflectColor = fakeCube(_R);
    #endif

    return reflectColor * frsnl;
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