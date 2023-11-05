/*
contributors: Patricio Gonzalez Vivo
description: calculate point light
use: lightPoint(<float3> _diffuseColor, <float3> _specularColor, <float3> _N, <float3> _V, <float> _NoV, <float> _f0, out <float3> _diffuse, out <float3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
    - LIGHT_FALLOFF: in glslViewer is u_lightFalloff
*/

#include "../specular.hlsl"
#include "../diffuse.hlsl"
#include "falloff.hlsl"

#ifndef SURFACE_POSITION
#define SURFACE_POSITION float(0.0, 0.0, 0.0)
#endif

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

#ifndef LIGHT_FALLOFF
#define LIGHT_FALLOFF   0.0
#endif

#ifndef FNC_LIGHT_POINT
#define FNC_LIGHT_POINT

void lightPoint(float3 _diffuseColor, float3 _specularColor, float3 _N, float3 _V, float _NoV, float _roughness, float _f0, float _shadow, inout float3 _diffuse, inout float3 _specular) {
    float3 toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    float toLightLength = length(toLight);
    float3 s = toLight/toLightLength;

    float NoL = dot(_N, s);

    float dif = diffuse(s, _N, _V, _NoV, NoL, _roughness);// * INV_PI;
    float spec = specular(s, _N, _V, _NoV, NoL, _roughness, _f0);

    float3 lightContribution = LIGHT_COLOR * LIGHT_INTENSITY * _shadow;
    #ifdef LIGHT_FALLOFF
    if (LIGHT_FALLOFF > 0.0)
        lightContribution *= falloff(toLightLength, LIGHT_FALLOFF);
    #endif

    _diffuse +=  _diffuseColor * lightContribution * dif;
    _specular += _specularColor * lightContribution * spec;
}

void lightPoint(float3 _diffuseColor, float3 _specularColor, float3 _N, float3 _V, float _NoV, float _roughness, float _f0, inout float3 _diffuse, inout float3 _specular) {
    lightPoint(_diffuseColor, _specularColor, _N, _V,  _NoV, _roughness, _f0, 1.0, _diffuse, _specular);
}

#endif