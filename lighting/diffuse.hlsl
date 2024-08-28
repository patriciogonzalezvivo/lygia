#include "diffuse/lambert.hlsl"
#include "diffuse/orenNayar.hlsl"
#include "diffuse/burley.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution
use: lightSpot(<float3> _diffuseColor, <float3> _specularColor, <float3> _N, <float3> _V, <float> _NoV, <float> _f0, out <float3> _diffuse, out <float3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DIFFUSE_FNC 
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define DIFFUSE_FNC diffuseLambert
#else
#define DIFFUSE_FNC diffuseOrenNayar
#endif  
#endif

#ifndef FNC_DIFFUSE
#define FNC_DIFFUSE
float diffuse(ShadingData shadingData) { return DIFFUSE_FNC(shadingData); }
#endif