#include "specular/phong.hlsl"
#include "specular/blinnPhong.hlsl"
#include "specular/cookTorrance.hlsl"
#include "specular/gaussian.hlsl"
#include "specular/beckmann.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate specular contribution
use:
    - specular(<float3> L, <float3> N, <float3> V, <float> roughness[, <float> fresnel])
    - specular(<float3> L, <float3> N, <float3> V, <float> NoV, <float> NoL, <float> roughness, <float> fresnel)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SPECULAR_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_FNC specularBlinnPhongRoughness
#else
#define SPECULAR_FNC specularCookTorrance
#endif  
#endif

#ifndef FNC_SPECULAR
#define FNC_SPECULAR
float3 specular(ShadingData shadingData) { return float3(1.0, 1.0, 1.0) * SPECULAR_FNC(shadingData); }
#endif