/*
contributors: Shadi El Hajj
description: Calculate indirect light
use: void lightIBLEvaluate(<Material> mat, inout <ShadingData> shadingData)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

#include "../envMap.glsl"
#include "../fresnelReflection.glsl"
#include "../sphericalHarmonics.glsl"
#include "../specular/importanceSampling.glsl"
#include "../reflection.glsl"
#include "../common/specularAO.glsl"
#include "../common/envBRDFApprox.glsl"
#include "../../color/tonemap.glsl"

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_LIGHT_IBL_EVALUATE
#define FNC_LIGHT_IBL_EVALUATE

void lightIBLEvaluate(Material mat, inout ShadingData shadingData) {

#if !defined(IBL_IMPORTANCE_SAMPLING) ||  __VERSION__ < 130 || defined(SCENE_SH_ARRAY)
    vec2 E = envBRDFApprox(shadingData.NoV, shadingData.roughness);    
    vec3 specularColorE = shadingData.specularColor * E.x + E.y;
#endif

    vec3 energyCompensation = vec3(1.0, 1.0, 1.0);

#if defined(IBL_IMPORTANCE_SAMPLING) &&  __VERSION__ >= 130
    vec3 Fr = specularImportanceSampling(shadingData.linearRoughness, shadingData.specularColor,
        mat.position, shadingData.N, shadingData.V, shadingData.R, shadingData.NoV, energyCompensation);
#else
    vec3 R = mix(shadingData.R, shadingData.N, shadingData.roughness*shadingData.roughness);
    vec3 Fr = envMap(R, shadingData.roughness, mat.metallic);
    Fr *= specularColorE;
#endif
    Fr *= energyCompensation;

#if !defined(PLATFORM_RPI) && defined(SHADING_MODEL_IRIDESCENCE)
    Fr  += fresnelReflection(mat, shadingData);
#endif

vec3 Fd = shadingData.diffuseColor;
#if defined(SCENE_SH_ARRAY)
    #ifdef GLSLVIEWER
    Fd *= tonemap(sphericalHarmonics(shadingData.N));
    #else
    Fd *= (sphericalHarmonics(shadingData.N));
    #endif
#else
    Fd *= envMap(shadingData.N, 1.0);
#endif

#if !defined(IBL_IMPORTANCE_SAMPLING)
    Fd *= (1.0-specularColorE);
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