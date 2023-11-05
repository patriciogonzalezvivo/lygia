#include "specular/phong.hlsl"
#include "specular/blinnPhong.hlsl"
#include "specular/cookTorrance.hlsl"
#include "specular/gaussian.hlsl"
#include "specular/beckmann.hlsl"
#include "specular/ggx.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: calculate specular contribution
use: 
    - specular(<float3> L, <float3> N, <float3> V, <float> roughness[, <float> fresnel])
    - specular(<float3> L, <float3> N, <float3> V, <float> NoV, <float> NoL, <float> roughness, <float> fresnel)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
*/

#ifndef SPECULAR_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_FNC specularBlinnPhongRoughnes
#else
#define SPECULAR_FNC specularCookTorrance
#endif  
#endif

#ifndef FNC_SPECULAR
#define FNC_SPECULAR
float specular(float3 L, float3 N, float3 V, float roughness) { return SPECULAR_FNC(L, N, V, roughness); }
float specular(float3 L, float3 N, float3 V, float roughness, float fresnel) { return SPECULAR_FNC(L, N, V, roughness, fresnel); }
float specular(float3 L, float3 N, float3 V, float NoV, float NoL, float roughness, float fresnel) { return SPECULAR_FNC(L, N, V, NoV, NoL, roughness, fresnel); }
#endif