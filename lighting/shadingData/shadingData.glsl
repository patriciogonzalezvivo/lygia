/*
contributors:  Shadi El Hajj
description: Structure to hold shading variables
*/

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
   float linearRoughness;
   vec3 diffuseColor;
   vec3 specularColor;

   vec3 diffuse;
   vec3 specular;
};
