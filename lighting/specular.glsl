#include "specular/phong.glsl"
#include "specular/blinnPhong.glsl"
#include "specular/cookTorrance.glsl"
#include "specular/gaussian.glsl"
#include "specular/beckmann.glsl"
#include "specular/ggx.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate specular contribution
use:
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> roughness[, <float> fresnel])
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness, <float> fresnel)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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
float specular(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness) { return SPECULAR_FNC(L, N, V, roughness); }
float specular(const in vec3 L, const in vec3 N, const in vec3 V, const in float roughness, const in float fresnel) { return SPECULAR_FNC(L, N, V, roughness, fresnel); }
float specular(const in vec3 L, const in vec3 N, const in vec3 V, const in float NoV, const in float NoL, const in float roughness, const in float fresnel) { return SPECULAR_FNC(L, N, V, NoV, NoL, roughness, fresnel); }
#endif