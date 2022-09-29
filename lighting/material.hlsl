#include "material/albedo.hlsl"
#include "material/specular.glsl"
#include "material/emissive.glsl"
#include "material/occlusion.glsl"

#include "material/normal.glsl"

#include "material/metallic.glsl"
#include "material/roughness.glsl"

#include "material/shininess.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Generic Material Structure
options:
    - SURFACE_POSITION
    - SHADING_SHADOWS
    - MATERIAL_CLEARCOAT_THICKNESS
    - MATERIAL_CLEARCOAT_ROUGHNESS
    - MATERIAL_CLEARCOAT_THICKNESS_NORMAL
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

    float3  position;       // world position of the surface
    float3  normal;         // world normal of the surface
    
    float3  f0;             // = float3(0.04);
    float   reflectance;    // = 0.5;

    float   roughness;
    float   metallic;
    float   ambientOcclusion;
    float   shadow;         // = 1.0;

#if defined(MATERIAL_CLEARCOAT_THICKNESS)
    float   clearCoat;
    float   clearCoatRoughness;
    #if defined(MATERIAL_CLEARCOAT_THICKNESS_NORMAL)
    float3  clearCoatNormal;// = float3(0.0, 0.0, 1.0);
    #endif
#endif

#if defined(SHADING_MODEL_SUBSURFACE)
    float   thickness;      // = 0.5;
    float   subsurfacePower;// = 12.234;
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