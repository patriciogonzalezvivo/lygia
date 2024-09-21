#include "shadingData.hlsl"
#include "../material.hlsl"
#include "../reflection.hlsl"
#include "../ior/reflectance2f0.hlsl"
#include "../common/perceptual2linearRoughness.hlsl"

/*
contributors:  Shadi El Hajj
description: ShadingData constructor
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_SHADING_DATA_NEW
#define FNC_SHADING_DATA_NEW 

ShadingData shadingDataNew() {
   ShadingData shadingData;

   shadingData.V = float3(0.0, 0.0, 0.0);
   shadingData.N = float3(0.0, 0.0, 0.0);
   shadingData.H = float3(0.0, 0.0, 0.0);
   shadingData.L = float3(0.0, 0.0, 0.0);
   shadingData.R = float3(0.0, 0.0, 0.0);

   shadingData.NoV = 0.0;
   shadingData.NoL = 0.0;
   shadingData.NoH = 0.0;

   shadingData.roughness = 0.0;
   shadingData.linearRoughness = 0.0;
   shadingData.diffuseColor = float3(0.0, 0.0, 0.0);
   shadingData.specularColor = float3(0.0, 0.0, 0.0);

   shadingData.energyCompensation = float3(1.0, 1.0, 1.0);

   shadingData.directDiffuse = float3(0.0, 0.0, 0.0);
   shadingData.directSpecular = float3(0.0, 0.0, 0.0);
   shadingData.indirectDiffuse = float3(0.0, 0.0, 0.0);
   shadingData.indirectSpecular = float3(0.0, 0.0, 0.0);

   return shadingData;
}

void shadingDataNew(Material mat, inout ShadingData shadingData) {
   float dielectricF0 = reflectance2f0(mat.reflectance);
   shadingData.N = mat.normal;
   shadingData.R = reflection(shadingData.V, shadingData.N, mat.roughness);
   shadingData.NoV = dot(shadingData.N, shadingData.V);
   shadingData.roughness = max(mat.roughness, MIN_PERCEPTUAL_ROUGHNESS);
   shadingData.linearRoughness = perceptual2linearRoughness(shadingData.roughness);
   shadingData.diffuseColor = mat.albedo.rgb * (1.0 - mat.metallic);
   shadingData.specularColor = lerp(float3(dielectricF0, dielectricF0, dielectricF0), mat.albedo.rgb, mat.metallic);
}

#endif
