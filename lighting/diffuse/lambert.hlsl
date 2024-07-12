#include "../../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using lambert equation
use:
    - <float> diffuseLambert(<float3> light, <float3> normal [, <float3> view, <float> roughness] )
    - <float> diffuseLambert(<float3> L, <float3> N, <float3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DIFFUSE_LAMBERT
#define FNC_DIFFUSE_LAMBERT
float diffuseLambert() { return INV_PI; }
float diffuseLambert(float3 L, float3 N) { return max(0.0, dot(N, L)); }
float diffuseLambert(float3 L, float3 N, float3 V, float roughness) { return diffuseLambert(L, N); }
float diffuseLambert(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness) { return diffuseLambert(L, N); }
#endif