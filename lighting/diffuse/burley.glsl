#include "../common/schlick.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: calculate diffuse contribution using burley equation
use: 
    - <float> diffuseBurley(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness] )
    - <float> diffuseBurley(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
*/

#ifndef FNC_DIFFUSE_BURLEY
#define FNC_DIFFUSE_BURLEY

float diffuseBurley(const in float NoV, const in float NoL, const in float LoH, const in float linearRoughness) {
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
    float lightScatter = schlick(1.0, f90, NoL);
    float viewScatter  = schlick(1.0, f90, NoV);
    return lightScatter * viewScatter;
}

float diffuseBurley(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness) {
    float LoH = max(dot(L, normalize(L + V)), 0.001);
    return diffuseBurley(NoV, NoL, LoH, roughness * roughness);
}

float diffuseBurley(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) {
    vec3 H = normalize(V + L);
    float NoV = clamp(dot(N, V), 0.001, 1.0);
    float NoL = clamp(dot(N, L), 0.001, 1.0);
    float LoH = clamp(dot(L, H), 0.001, 1.0);

    return diffuseBurley(NoV, NoL, LoH, roughness * roughness);
}

#endif