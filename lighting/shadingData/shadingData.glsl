/*
contributors:  Shadi El Hajj
description: Structure to hold shading variables
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef STR_SHADING_DATA
#define STR_SHADING_DATA

struct ShadingData {
   vec3 V;
   vec3 N;
   vec3 H;
   vec3 L;
   vec3 R;

   float NoV;
   float NoL;
   float NoH;

   float roughness;
   float linearRoughness;
   vec3 diffuseColor;
   vec3 specularColor;

   vec3 energyCompensation;

   vec3 directDiffuse;
   vec3 directSpecular;
   vec3 indirectDiffuse;
   vec3 indirectSpecular;
};

#endif