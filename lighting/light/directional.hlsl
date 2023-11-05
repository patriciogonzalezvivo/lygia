#include "../specular.hlsl"
#include "../diffuse.hlsl"
#include "falloff.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: calculate directional light
use: lightDirectional(<float3> _diffuseColor, <float3> _specularColor, <float3> _N, <float3> _V, <float> _NoV, <float> _f0, out <float3> _diffuse, out <float3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: Position
    - LIGHT_DIRECTION
    - LIGHT_COLOR: Color
    - LIGHT_INTENSITY: Intensity
*/

#ifndef LIGHT_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define LIGHT_POSITION _WorldSpaceLightPos0.xyz
#else
#define LIGHT_POSITION  float3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(UNITY_COMPILER_HLSL)
#include <UnityLightingCommon.cginc>
#define LIGHT_COLOR     _LightColor0.rgb
#else
#define LIGHT_COLOR     float3(0.5, 0.5, 0.5)
#endif
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef FNC_LIGHT_DIRECTIONAL
#define FNC_LIGHT_DIRECTIONAL

void lightDirectional(float3 _diffuseColor, float3 _specularColor, float3 _N, float3 _V, float _NoV, float _roughness, float _f0, float _shadow, inout float3 _diffuse, inout float3 _specular) {
    #ifdef LIGHT_DIRECTION
    float3 s = normalize(LIGHT_DIRECTION);
    #else 
    float3 s = normalize(LIGHT_POSITION);
    #endif
    float NoL = dot(_N, s);
    float dif = diffuseOrenNayar(s, _N, _V, _NoV, NoL, _roughness);
    float spec = specularCookTorrance(s, _N, _V, _NoV, NoL, _roughness, _f0);
    _diffuse += max(0.0, LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif) * _shadow);
    _specular += max(0.0, LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec) * _shadow);
}

// void lightDirectional(float3 _diffuseColor, float3 _specularColor, float3 _N, float3 _V, float _NoV, float _roughness, float _f0, inout float3 _diffuse, inout float3 _specular) {
//     return lightDirectional(_diffuseColor, _specularColor, _N, _V, _NoV, _roughness, _f0, 1.0, _diffuse, _specular);
// }

#endif