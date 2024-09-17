/*
contributors: Shadi El Hajj
description: Calculate indirect light
use: void lightIBLEvaluate(<Material> mat, inout <ShadingData> shadingData)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

#include "../envMap.hlsl"
#include "../fresnelReflection.hlsl"
#include "../sphericalHarmonics.hlsl"
#include "../specular/importanceSampling.hlsl"
#include "../reflection.hlsl"
#include "../common/specularAO.hlsl"
#include "../common/envBRDFApprox.hlsl"

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_LIGHT_IBL_EVALUATE
#define FNC_LIGHT_IBL_EVALUATE

void lightIBLEvaluate(Material mat, inout ShadingData shadingData) {

#if !defined(IBL_IMPORTANCE_SAMPLING) || defined(SCENE_SH_ARRAY)
    float2 E = envBRDFApprox(shadingData.NoV, shadingData.roughness);    
    float3 specularColorE = shadingData.specularColor * E.x + E.y;
#endif

float3 energyCompensation = float3(1.0, 1.0, 1.0);

#if defined(IBL_IMPORTANCE_SAMPLING)
    float3 Fr = specularImportanceSampling(shadingData.linearRoughness, shadingData.specularColor,
        mat.position, shadingData.N, shadingData.V, shadingData.R, shadingData.NoV, energyCompensation);
#else
    float3 R = lerp(shadingData.R, shadingData.N, shadingData.roughness*shadingData.roughness);
    float3 Fr = envMap(R, shadingData.roughness, mat.metallic);
    Fr *= specularColorE;
#endif
    Fr *= energyCompensation;

#if !defined(PLATFORM_RPI) && defined(SHADING_MODEL_IRIDESCENCE)
    Fr  += fresnelReflection(mat, shadingData);
#endif

#if defined(SCENE_SH_ARRAY)
    float3 Fd = shadingData.diffuseColor * (1.0-specularColorE);
    Fd  *= sphericalHarmonics(shadingData.N);
#elif defined(IBL_IMPORTANCE_SAMPLING)
    float3 Fd = shadingData.diffuseColor;
    Fd *= envMap(shadingData.N, 1.0);
#else
    float3 Fd = shadingData.diffuseColor * (1.0-specularColorE);
    Fd *= envMap(shadingData.N, 1.0);
#endif

    // AO
    float diffuseAO = mat.ambientOcclusion;
    Fd  *= diffuseAO;
    Fr  *= specularAO(mat, shadingData, diffuseAO);

    shadingData.energyCompensation = energyCompensation;

    shadingData.indirectDiffuse = Fd * IBL_LUMINANCE;
    shadingData.indirectSpecular = Fr * IBL_LUMINANCE;
}

#endif