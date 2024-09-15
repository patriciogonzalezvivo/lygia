#include "../../math/const.glsl"
#include "../shadingData/shadingData.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using lambert equation
use:
    - <float> diffuseLambert(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness])
    - <float> diffuseLambert(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DIFFUSE_LAMBERT
#define FNC_DIFFUSE_LAMBERT
float diffuseLambertConstant() { return INV_PI; }
float diffuseLambertConstant(ShadingData shadingData) { return diffuseLambertConstant(); }

float diffuseLambert(const in vec3 L, const in vec3 N) { return max(0.0, dot(N, L)); }
float diffuseLambert(ShadingData shadingData) { return max(0.0, shadingData.NoL); }
#endif