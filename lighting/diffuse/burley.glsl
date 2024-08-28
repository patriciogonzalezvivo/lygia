#include "../common/schlick.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using burley equation
use:
    - <float> diffuseBurley(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness])
    - <float> diffuseBurley(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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

float diffuseBurley(ShadingData shadingData) {
    float LoH = dot(shadingData.L, shadingData.H);
    return diffuseBurley(shadingData.NoV, shadingData.NoL, LoH, shadingData.linearRoughness);
}

#endif