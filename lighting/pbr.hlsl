#ifndef CAMERA_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define CAMERA_POSITION _WorldSpaceCameraPos
#else
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif
#endif

#ifndef LIGHT_DIRECTION
#if defined(UNITY_COMPILER_HLSL)
#define LIGHT_DIRECTION _WorldSpaceLightPos0.xyz
#else
#define LIGHT_DIRECTION float3(0.0, 10.0, -50.0)
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

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#include "../color/tonemap.hlsl"

#include "material.hlsl"
#include "envMap.hlsl"
#include "fresnelReflection.hlsl"
#include "sphericalHarmonics.hlsl"
#include "light/new.hlsl"
#include "light/resolve.hlsl"

#include "reflection.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use: <float4> pbr( <Material> _material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR in glslViewer is u_lightColor
    - CAMERA_POSITION: in glslViewer is u_camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_PBR
#define FNC_PBR

float4 pbr(const Material _mat) {
    // Calculate Color
    float3 diffuseColor = _mat.albedo.rgb * (float3(1.0, 1.0, 1.0) - _mat.f0) * (1.0 - _mat.metallic);
    float3 specularColor = lerp(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    // Cached
    Material M = _mat;
    M.V = normalize(CAMERA_POSITION - M.position); // View
    M.NoV = dot(M.normal, M.V); // Normal . View
    M.R = reflection(M.V, M.normal, M.roughness); // Reflection

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     vec2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 

    // Global Ilumination ( Image Based Lighting )
    // ------------------------
    float3 E = envBRDFApprox(specularColor, M);
    float diffuseAO = min(M.ambientOcclusion, ssao);
    
    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr = envMap(M) * E;
    #if !defined(PLATFORM_RPI)
    Fr  += fresnelReflection(M);
    #endif
    Fr  *= specularAO(M, diffuseAO);

    float3 Fd = diffuseColor;
    #if defined(UNITY_COMPILER_HLSL)
    Fd *= ShadeSH9(half4(M.normal,1));
    #elif defined(SCENE_SH_ARRAY)
    Fd  *= tonemap( sphericalHarmonics(M.normal) );
    #endif
    Fd  *= diffuseAO;
    Fd  *= (1.0 - E);

    // Local Ilumination
    // ------------------------
    float3 lightDiffuse = float3(0.0, 0.0, 0.0);
    float3 lightSpecular = float3(0.0, 0.0, 0.0);
    
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);
        #endif

        #if defined(LIGHT_POINTS) && defined(LIGHT_POINTS_TOTAL)
        for (int i = 0; i < LIGHT_POINTS_TOTAL; i++) {
            LightPoint L = LIGHT_POINTS[i];
            lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);
        }
        #endif
    }
    
    // Final Sum
    // ------------------------
    float4 color = float4(0.0, 0.0, 0.0, 1.0);

    // Diffuse
    color.rgb += Fd * IBL_LUMINANCE;
    color.rgb += lightDiffuse;

    // Specular
    color.rgb += Fr * IBL_LUMINANCE;
    color.rgb += lightSpecular;
    color.rgb *= M.ambientOcclusion;
    color.rgb += M.emissive;
    color.a = M.albedo.a;

    return color;
}
#endif
