#include "../specular.hlsl"
#include "../diffuse.hlsl"
#include "falloff.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate directional light
use: lightDirectional(<float3> _diffuseColor, <float3> _specularColor, <float3> _N, <float3> _V, <float> _NoV, <float> _f0, out <float3> _diffuse, out <float3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: Position
    - LIGHT_DIRECTION
    - LIGHT_COLOR: Color
    - LIGHT_INTENSITY: Intensity
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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

#ifndef STR_LIGHT_DIRECTIONAL
#define STR_LIGHT_DIRECTIONAL
struct LightDirectional
{
    float3 direction;
    float3 color;
    float intensity;
};
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

void lightDirectional(
    const in float3 _diffuseColor, const in float3 _specularColor,
    const in float3 _V,
    const in float3 _Ld, const in float3 _Lc, const in float _Li,
    const in float3 _N, const in float _NoV, const in float _NoL,
    const in float _roughness, const in float _f0,
    inout float3 _diffuse, inout float3 _specular)
{
    float dif = diffuse(_Ld, _N, _V, _NoV, _NoL, _roughness);
    float spec = specular(_Ld, _N, _V, _NoV, _NoL, _roughness, _f0);

    _diffuse += max(float3(0.0, 0.0, 0.0), _Li * (_diffuseColor * _Lc * dif));
    _specular += max(float3(0.0, 0.0, 0.0), _Li * (_specularColor * _Lc * spec));
}

#ifdef STR_MATERIAL
void lightDirectional(
    const in float3 _diffuseColor, const in float3 _specularColor,
    LightDirectional _L, const in Material _mat, 
    inout float3 _diffuse, inout float3 _specular) {

    float f0    = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
    float NoL   = dot(_mat.normal, _L.direction);

    lightDirectional(
        _diffuseColor, _specularColor, 
        _mat.V, 
        _L.direction, _L.color, _L.intensity,
        _mat.normal, _mat.NoV, NoL, _mat.roughness, f0, 
        _diffuse, _specular);

#ifdef SHADING_MODEL_SUBSURFACE
    float3  h     = normalize(_mat.V + _L.direction);
    float NoH   = saturate(dot(_mat.normal, h));
    float LoH   = saturate(dot(_L.direction, h));

    float scatterVoH = saturate(dot(_mat.V, -_L.direction));
    float forwardScatter = exp2(scatterVoH * _mat.subsurfacePower - _mat.subsurfacePower);
    float backScatter = saturate(NoL * _mat.subsurfaceThickness + (1.0 - _mat.subsurfaceThickness)) * 0.5;
    float subsurface = lerp(backScatter, 1.0, forwardScatter) * (1.0 - _mat.subsurfaceThickness);
    _diffuse += _mat.subsurfaceColor * (subsurface * diffuseLambert());
#endif
}
#endif

#endif