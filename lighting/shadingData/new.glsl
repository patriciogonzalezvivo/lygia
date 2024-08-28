#include "shadingData.glsl"
#include "../reflection.glsl"
#include "../ior/reflectance2f0.glsl"
#include "../common/perceptual2LinearRoughness.glsl"

/*
contributors:  Shadi El Hajj
description: ShadingData constructor
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_SHADING_DATA_NEW
#define FNC_SHADING_DATA_NEW 

ShadingData shadingDataNew() {
   ShadingData shadingData;

   shadingData.V = vec3(0.0, 0.0, 0.0);
   shadingData.N = vec3(0.0, 0.0, 0.0);
   shadingData.H = vec3(0.0, 0.0, 0.0);
   shadingData.L = vec3(0.0, 0.0, 0.0);
   shadingData.R = vec3(0.0, 0.0, 0.0);

   shadingData.NoV = 0.0;
   shadingData.NoL = 0.0;
   shadingData.NoH = 0.0;

   shadingData.roughness = 0.0;
   shadingData.linearRoughness = 0.0;
   shadingData.diffuseColor = vec3(0.0, 0.0, 0.0);
   shadingData.specularColor = vec3(0.0, 0.0, 0.0);

   shadingData.diffuse = vec3(0.0, 0.0, 0.0);
   shadingData.specular = vec3(0.0, 0.0, 0.0);

   return shadingData;
}

void shadingDataNew(Material mat, inout ShadingData shadingData) {
   float dielectricF0 = reflectance2f0(mat.reflectance);
   shadingData.N = mat.normal;
   shadingData.R = reflection(shadingData.V, shadingData.N, mat.roughness);
   shadingData.NoV = dot(shadingData.N, shadingData.V);
   shadingData.roughness = mat.roughness;
   shadingData.linearRoughness = perceptual2LinearRoughness(shadingData.roughness);
   shadingData.diffuseColor = mat.albedo.rgb * (1.0 - mat.metallic);
   shadingData.specularColor = mix(vec3(dielectricF0, dielectricF0, dielectricF0), mat.albedo.rgb, mat.metallic);
}

#endif
