#include "shadingData/new.hlsl"
#include "material/roughness.hlsl"
#include "material/normal.hlsl"
#include "material/albedo.hlsl"
#include "material.hlsl"
#include "light/new.hlsl"
#include "specular.hlsl"
#include "diffuse.hlsl"
#include "reflection.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Render with a gooch stylistic shading model
use: <float4> gooch(<float4> albedo, <float3> normal, <float3> light, <float3> view, <float> roughness)
options:
    - GOOCH_WARM: default float3(0.25, 0.15, 0.0)
    - GOOCH_COLD: default float3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: default float3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD: in glslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP: in glslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in glslViewer is 1024.0
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define CAMERA_POSITION _WorldSpaceCameraPos
#else
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif
#endif

#ifndef LIGHT_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define LIGHT_POSITION _WorldSpaceLightPos0.xyz
#else
#define LIGHT_POSITION  float3(0.0, 10.0, -50.0)
#endif
#endif



#ifndef GOOCH_WARM 
#define GOOCH_WARM float3(0.25, 0.15, 0.0)
#endif 

#ifndef GOOCH_COLD 
#define GOOCH_COLD float3(0.0, 0.0, 0.2)
#endif 

#ifndef GOOCH_SPECULAR
#if defined(UNITY_COMPILER_HLSL)
#include <UnityLightingCommon.cginc>
#define GOOCH_SPECULAR     _LightColor0.rgb
#else
#define GOOCH_SPECULAR     float3(1.0, 1.0, 1.0)
#endif
#endif

#ifndef FNC_GOOCH
#define FNC_GOOCH
float4 gooch(const in float4 _albedo, const in float3 _N, const in float3 _L, const in float3 _V, const in float _roughness, const in float _Li) {
    float3 warm = GOOCH_WARM + _albedo.rgb * 0.6;
    float3 cold = GOOCH_COLD + _albedo.rgb * 0.1;

    ShadingData shadingData = shadingDataNew();
    shadingData.L = normalize(_L);
    shadingData.N = normalize(_N);
    shadingData.V = normalize(_V);
    shadingData.H = normalize(shadingData.L + shadingData.V);
    shadingData.NoV = dot(shadingData.N, shadingData.V);
    shadingData.NoL = dot(shadingData.N, shadingData.L);
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));
    shadingData.roughness = _roughness;

    // Lambert Diffuse
    float diff = diffuse(shadingData) * _Li;
    // Phong Specular
    float3 spec = float3(1.0, 1.0, 1.0) * specularBlinnPhongRoughness(shadingData) * _Li;

    return float4(lerp(lerp(cold, warm, diff), GOOCH_SPECULAR, spec), _albedo.a);
}

float4 gooch(const in LightDirectional _L, in Material _M, ShadingData shadingData) {
    return gooch(_M.albedo, _M.normal, _L.direction, shadingData.V, _M.roughness, _L.intensity);
}

float4 gooch(const in LightPoint _L, in Material _M, ShadingData shadingData) {
    return gooch(_M.albedo, _M.normal, _L.position, shadingData.V, _M.roughness, _L.intensity);
}

float4 gooch(const in Material _M, ShadingData shadingData) {
    #if defined(LIGHT_DIRECTION)
    LightDirectional L;
    #elif defined(LIGHT_POSITION)
    LightPoint L;
    #endif
    lightNew(L);

    #if defined(FNC_RAYMARCH_SOFTSHADOW)
    #if defined(LIGHT_DIRECTION)
    L.intensity *= raymarchSoftShadow(_M.position, L.direction);
    #elif defined(LIGHT_POSITION)
    L.intensity *= raymarchSoftShadow(_M.position, L.position);
    #endif
    #endif 

    return gooch(L, _M, shadingData) * _M.ambientOcclusion;
}

float4 gooch(const in Material _M) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - _M.position);
    return gooch(_M, shadingData);
}

#endif