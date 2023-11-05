#include "material/albedo.hlsl"
#include "material/specular.hlsl"
#include "material/emissive.hlsl"
#include "material/occlusion.hlsl"

#include "material/normal.hlsl"

#include "material/metallic.hlsl"
#include "material/roughness.hlsl"

#include "material/shininess.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Generic Material Structure
options:
    - SURFACE_POSITION
    - SHADING_SHADOWS
    - MATERIAL_HAS_CLEAR_COAT
    - MATERIAL_CLEARCOAT_ROUGHNESS
    - MATERIAL_HAS_CLEAR_COAT_NORMAL
    - SHADING_MODEL_SUBSURFACE
    - MATERIAL_SUBSURFACE_COLOR
    - SHADING_MODEL_CLOTH
    - SHADING_MODEL_SPECULAR_GLOSSINESS
*/

#ifndef STR_MATERIAL
#define STR_MATERIAL
struct Material {
    float4  albedo;
    float3  emissive;

    float3  position;           // world position of the surface
    float3  normal;             // world normal of the surface

    #if defined(SCENE_BACK_SURFACE)
    float3  normal_back;        // world normal of the back surface of the model
    #endif
    
    float3  ior;                // Index of Refraction
    float3  f0;                 // reflectance at 0 degree

    float   roughness;
    float   metallic;
    float   ambientOcclusion;   // default 1.0
    float   shadow;             // default 1.0

    float   clearCoat;
    float   clearCoatRoughness;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    float3  clearCoatNormal;    // default float3(0.0, 0.0, 1.0)
    #endif

#if defined(SHADING_MODEL_SUBSURFACE) || defined(SCENE_BACK_SURFACE)
    float   thickness;          // default 0.5;
    float   subsurfacePower;    // default 12.234;
#endif

#if defined(SHADING_MODEL_CLOTH)
    float3  sheenColor;
#endif

#if defined(MATERIAL_SUBSURFACE_COLOR)
    float3  subsurfaceColor;// = float3(1.0);
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    float3  specularColor;
    float   glossiness;
#endif

};
#endif