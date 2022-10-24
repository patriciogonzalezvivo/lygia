#include "diffuse/lambert.hlsl"
#include "diffuse/orenNayar.hlsl"
#include "diffuse/burley.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: calculate diffuse contribution
use: lightSpot(<float3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
*/

#ifndef DIFFUSE_FNC 
#define DIFFUSE_FNC diffuseLambert
#endif

#ifndef FNC_DIFFUSE
#define FNC_DIFFUSE
float diffuse(float3 _L, float3 _N, float3 _V, float _roughness) { return DIFFUSE_FNC(_L, _N, _V, _roughness); }
float diffuse(float3 _L, float3 _N, float3 _V, float _NoV, float _NoL, float _roughness) { return DIFFUSE_FNC(_L, _N, _V, _NoV, _NoL, _roughness); }
#endif