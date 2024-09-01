#ifndef DIFFUSE_FNC
#define DIFFUSE_FNC diffuseLambertConstant
#endif

#ifndef SPECULAR_FNC
#define SPECULAR_FNC specularCookTorrance
#endif

#include "shadingData/new.hlsl"
#include "material.hlsl"
#include "envMap.hlsl"
#include "fresnelReflection.hlsl"
#include "sphericalHarmonics.hlsl"
#include "light/new.hlsl"
#include "light/resolve.hlsl"

#include "reflection.hlsl"
#include "diffuse/importanceSampling.hlsl"
#include "specular/importanceSampling.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"

/*
contributors: [Patricio Gonzalez Vivo, Shadi El Hajj]
description: Simple PBR shading model
use: <float4> pbr( <Material> material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
    - RAYMARCH_AO: enabled raymarched ambient occlusion
examples:
    - /shaders/lighting_raymarching_pbr.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBR
#define FNC_PBR

float4 pbr(const Material mat, ShadingData shadingData) {
    shadingDataNew(mat, shadingData);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    float2 E = envBRDFApprox(shadingData.NoV, shadingData.roughness);    
    float3 specularColorE = shadingData.specularColor * E.x + E.y;

#if defined(IBL_IMPORTANCE_SAMPLING)
    float3 Fr = specularImportanceSampling(shadingData.linearRoughness, shadingData.specularColor, shadingData.N, shadingData.V, shadingData.R, shadingData.NoV);
#else
    float3 Fr = envMap(mat, shadingData) * specularColorE;
#endif

    #if !defined(PLATFORM_RPI) && defined(SHADING_MODEL_IRIDESCENCE)
    Fr  += fresnelReflection(mat, shadingData);
    #endif

    float3 Fd = shadingData.diffuseColor;
#if defined(SCENE_SH_ARRAY)
    Fd  *= sphericalHarmonics(shadingData.N);
#elif defined(IBL_IMPORTANCE_SAMPLING)
    Fd *= diffuseImportanceSampling(shadingData.linearRoughness, shadingData.N, shadingData.V, shadingData.R);
#else
    Fd *= envMap(shadingData.N, 1.0);
#endif

    // AO
    float diffuseAO = mat.ambientOcclusion;
    Fd  *= diffuseAO;
    Fr  *= specularAO(mat, shadingData, diffuseAO);

    // Direct Lights
    // -------------

    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        lightResolve(L, mat, shadingData);
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        lightResolve(L, mat, shadingData);
        #endif

        #if defined(LIGHT_POINTS) && defined(LIGHT_POINTS_TOTAL)
        for (int i = 0; i < LIGHT_POINTS_TOTAL; i++) {
            LightPoint L = LIGHT_POINTS[i];
            lightResolve(L, mat, shadingData);
        }
        #endif
    }

    
    // Final Sum
    // ------------------------
    float4 color  = float4(0.0, 0.0, 0.0, 1.0);

    // Diffuse
    color.rgb  += Fd * IBL_LUMINANCE;
    color.rgb  += shadingData.diffuse;

    // Specular
    color.rgb  += Fr * IBL_LUMINANCE;
    color.rgb  += shadingData.specular; 
    color.rgb  += mat.emissive;
    color.a     = mat.albedo.a;

    return color;
}

float4 pbr(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbr(mat, shadingData);
}

#endif
