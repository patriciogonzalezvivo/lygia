#include "../color/tonemap.hlsl"

#include "material.hlsl"
#include "light/new.hlsl"
#include "envMap.hlsl"
#include "specular.hlsl"
#include "fresnelReflection.hlsl"
#include "transparent.hlsl"

#include "ior/2eta.hlsl"
#include "ior/2f0.hlsl"

#include "reflection.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple glass shading model
use:
    - <float4> glass(<Material> material)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - SCENE_BACK_SURFACE: null
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION: null
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
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

#ifndef FNC_PBRGLASS
#define FNC_PBRGLASS

float4 pbrGlass(const Material _mat) {
    
    // Cached
    Material M  = _mat;
    M.V         = normalize(CAMERA_POSITION - M.position);  // View
    M.R         = reflection(M.V, M.normal, M.roughness);   // Reflection
#if defined(SCENE_BACK_SURFACE)
    float3 No     = normalize(M.normal - M.normal_back); // Normal out is the difference between the front and back normals
#else
    float3 No     = M.normal;                            // Normal out
#endif
    M.NoV       = dot(No, M.V);                        // Normal . View

    float3 eta    = ior2eta(M.ior);
    

    // Global Ilumination ( Image Based Lighting )
    // ------------------------
    float3 E = envBRDFApprox(M.albedo.rgb, M);

    float3 Gi = float3(0.0, 0.0, 0.0);
    Gi  += envMap(M) * E;
    #if !defined(PLATFORM_RPI)
    // Gi  += fresnelReflection(M);

    #if defined(SHADING_MODEL_IRIDESCENCE)
    float3 Fr = float3(0.0, 0.0, 0.0);
    Gi  += fresnelIridescentReflection(M.normal, -M.V, M.f0, float3(IOR_AIR, IOR_AIR, IOR_AIR), M.ior, M.thickness, M.roughness, Fr);
    #else
    float3 Fr = fresnel(M.f0, M.NoV);
    Gi  += fresnelReflection(M.R, Fr) * (1.0-M.roughness);
    #endif

    #endif

    float4 color  = float4(0.0, 0.0, 0.0, 1.0);

    // Refraction
    color.rgb   += transparent(No, -M.V, Fr, eta, M.roughness);
    color.rgb   += Gi * IBL_LUMINANCE;

    // TODO: RaG
    //  - Add support for multiple lights
    // 
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        #endif

        #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
        // lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);
        float spec = specular(L.direction, M.normal, M.V, M.roughness);

        color.rgb += L.color * spec;

        #endif
    }

    return color;
}

#endif