#include "shadingData.hlsl"

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

   shadingData.fresnel = 0.0;
   shadingData.roughness = 0.0;
   shadingData.linearRoughness = 0.0;
   shadingData.diffuseColor = float3(0.0, 0.0, 0.0);
   shadingData.specularColor = float3(0.0, 0.0, 0.0);

   shadingData.diffuse = float3(0.0, 0.0, 0.0);
   shadingData.specular = float3(0.0, 0.0, 0.0);

   return shadingData;
}

#endif
