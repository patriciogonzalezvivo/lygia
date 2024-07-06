#include "common/schlick.glsl"

#include "../math/pow5.glsl"
#include "../math/const.glsl"
#include "../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Resolve fresnel coeficient
use:
    - <float|vec3> fresnel(const <float|vec3> f0, <float> NoV)
    - <float|vec3> fresnel(const <float|vec3> f0, <float> NoV, float roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const in vec3 f0, vec3 normal, vec3 view) {
   return schlick(f0, 1.0, dot(view, normal));
}

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

// Roughness-adjusted fresnel function to attenuate high speculars at glancing angles
// Very useful when used with filtered environment maps
// See https://seblagarde.wordpress.com/2011/08/17/hello-world/
vec3 fresnel(vec3 f0, float NoV, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow5(1.0 - NoV);
}

#endif