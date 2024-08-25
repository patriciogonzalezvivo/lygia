/*
contributors:  Shadi El Hajj
description: Structure to hold shading variables
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

   float fresnel;
   float roughness;
   float linearRoughness;
   vec3 diffuseColor;
   vec3 specularColor;

   vec3 diffuse;
   vec3 specular;
};

#endif