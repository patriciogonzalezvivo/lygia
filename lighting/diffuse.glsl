#include "diffuse/lambert.glsl"
#include "diffuse/orenNayar.glsl"
#include "diffuse/burley.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: calculate diffuse contribution
use: lightSpot(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
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
float diffuse(const in vec3 _L, const in vec3 _N, const in vec3 _V, const in float _roughness) { return DIFFUSE_FNC(_L, _N, _V, _roughness); }
float diffuse(const in vec3 _L, const in vec3 _N, const in vec3 _V, const in float _NoV, const in float _NoL, const in float _roughness) { return DIFFUSE_FNC(_L, _N, _V, _NoV, _NoL, _roughness); }
#endif