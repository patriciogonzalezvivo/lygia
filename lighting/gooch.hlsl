#include "material/roughness.hlsl"
#include "material/normal.hlsl"
#include "material/albedo.hlsl"

#include "diffuse.hlsl"
#include "specular.hlsl"

#include "material.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Render with a gooch stylistic shading model
use: <float4> gooch(<float4> albedo, <float3> normal, <float3> light, <float3> view, <float> roughness)
options:
    - GOOCH_WARM: defualt float3(0.25, 0.15, 0.0)
    - GOOCH_COLD: defualt float3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: defualt float3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD: in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP: in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0
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
float4 gooch(float4 albedo, float3 normal, float3 light, float3 view, float roughness, float shadow) {
    float3 warm = GOOCH_WARM + albedo.rgb * 0.6;
    float3 cold = GOOCH_COLD + albedo.rgb * 0.1;

    float3 l = normalize(light);
    float3 n = normalize(normal);
    float3 v = normalize(view);

    // Lambert Diffuse
    float diff = diffuse(l, n, v, roughness) * shadow;
    // Phong Specular
    float spec = specular(l, n, v, roughness) * shadow;

    return float4(lerp(lerp(cold, warm, diff), GOOCH_SPECULAR, spec), albedo.a);
}

float4 gooch(float4 albedo, float3 normal, float3 light, float3 view, float roughness) {
    return gooch(albedo, normal, light, view, roughness, 1.0);
}

float4 gooch(Material material) {
    #ifdef LIGHT_DIRECTION
    return gooch(material.albedo, material.normal, LIGHT_DIRECTION, (CAMERA_POSITION - material.position), material.roughness, material.shadow);
    #else
    return gooch(material.albedo, material.normal, (LIGHT_POSITION - material.position), (CAMERA_POSITION - material.position), material.roughness, material.shadow);
    #endif
}

#endif