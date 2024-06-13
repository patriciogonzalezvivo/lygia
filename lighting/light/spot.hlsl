/*
contributors: Patricio Gonzalez Vivo
description: Calculate spot light
use: lightSpot(<float3> _diffuseColor, <float3> _specularColor, <float3> _N, <float3> _V, <float> _NoV, <float> _f0, out <float3> _diffuse, out <float3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "../specular.hlsl"
#include "../diffuse.hlsl"
#include "falloff.hlsl"

#ifndef SURFACE_POSITION
#define SURFACE_POSITION float3(0.0, 0.0, 0.0)
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

#ifndef FNC_LIGHT_SPOT
#define FNC_LIGHT_SPOT

void lightSpot(float3 _diffuseColor, float3 _specularColor, float3 _N, float3 _V, float _NoV, float _roughness, float _f0, out float3 _diffuse, out float3 _specular) {
    float3 toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    float toLightLength = length(toLight);
    float3 s = toLight/toLightLength;

    float angle = acos(dot(-s, light.direction));
    float cutoff1 = radians(clamp(light.spotLightCutoff - max(light.spotLightFactor, 0.01), 0.0, 89.9));
    float cutoff2 = radians(clamp(light.spotLightCutoff, 0.0, 90.0));
    if (angle < cutoff2) {
        float dif = diffuseOrenNayar(s, _N, _V, _NoV, _roughness);
        float fall = falloff(toLightLength, light.spotLightDistance);
        float spec = specularCookTorrance(s, _N, _V, _NoV, _roughness);
        _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif * fall) * smoothstep(cutoff2, cutoff1, angle);
        _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec * fall) * smoothstep(cutoff2, cutoff1, angle);
    }
    else {
        _diffuse = float3(0.0, 0.0, 0.0);
        _specular = float3(0.0, 0.0, 0.0);
    }
}

#endif