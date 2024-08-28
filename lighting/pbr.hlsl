#include "../color/tonemap.hlsl"

#include "shadingData/new.hlsl"
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
    shadingData(mat, shadingData);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    float3 E = envBRDFApprox(shadingData);
    float diffuseAO = mat.ambientOcclusion;

    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr  = envMap(mat, shadingData) * E;
    #if !defined(PLATFORM_RPI)
    Fr  += fresnelReflection(mat, shadingData);
    #endif
    Fr  *= specularAO(mat, shadingData, diffuseAO);

    float3 Fd = shadingData.diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd  *= tonemap( sphericalHarmonics(shadingData.N) );
    #else
    Fd *= envMap(shadingData.N, 1.0);
    #endif
    Fd  *= diffuseAO;
    Fd  *= (1.0 - E);

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
