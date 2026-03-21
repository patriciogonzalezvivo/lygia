/*
contributors: Shadi El Hajj
description: Calculate indirect light
use: void lightIBLEvaluate(<Material> mat, inout <ShadingData> shadingData)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

#include "../envMap.wgsl"
#include "../fresnelReflection.wgsl"
#include "../sphericalHarmonics.wgsl"
#include "../specular/importanceSampling.wgsl"
#include "../reflection.wgsl"
#include "../common/specularAO.wgsl"
#include "../common/envBRDFApprox.wgsl"
#include "../../color/tonemap.wgsl"

const IBL_LUMINANCE: f32 = 1.0;

fn lightIBLEvaluate(mat: Material, shadingData: ShadingData) {

    let E = envBRDFApprox(shadingData.NoV, shadingData.roughness);
    let specularColorE = shadingData.specularColor * E.x + E.y;

    let energyCompensation = vec3f(1.0, 1.0, 1.0);

    vec3 Fr = specularImportanceSampling(shadingData.linearRoughness, shadingData.specularColor,
        mat.position, shadingData.N, shadingData.V, shadingData.R, shadingData.NoV, energyCompensation);
    let R = mix(shadingData.R, shadingData.N, shadingData.roughness*shadingData.roughness);
    let Fr = envMap(R, shadingData.roughness, mat.metallic);
    Fr *= specularColorE;
    Fr *= energyCompensation;

    Fr  += fresnelReflection(mat, shadingData);

let Fd = shadingData.diffuseColor;
    Fd *= tonemap(sphericalHarmonics(shadingData.N));
    Fd *= (sphericalHarmonics(shadingData.N));
    Fd *= envMap(shadingData.N, 1.0);

    Fd *= (1.0-specularColorE);

    // AO
    let diffuseAO = mat.ambientOcclusion;
    Fd  *= diffuseAO;
    Fr  *= specularAO(mat, shadingData, diffuseAO);

    shadingData.energyCompensation = energyCompensation;

    shadingData.indirectDiffuse = Fd * IBL_LUMINANCE;
    shadingData.indirectSpecular = Fr * IBL_LUMINANCE;
}
