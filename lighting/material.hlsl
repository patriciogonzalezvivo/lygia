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
    - SCENE_BACK_SURFACE
    - SHADING_MODEL_CLEAR_COAT
    - MATERIAL_HAS_CLEAR_COAT_NORMAL
    - SHADING_MODEL_IRIDESCENCE
    - SHADING_MODEL_SUBSURFACE
    - SHADING_MODEL_CLOTH
    - SHADING_MODEL_SPECULAR_GLOSSINESS
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if !defined(MATERIAL_OPT_IN)
#define RENDER_RAYMARCHING
#define SHADING_MODEL_CLEAR_COAT
#endif

#ifndef STR_MATERIAL
#define STR_MATERIAL
struct Material
{
    float4 albedo;
    float3 emissive;

    float3 position; // world position of the surface
    float3 normal;   // world normal of the surface

#if defined(RENDER_RAYMARCHING)
    float   sdf;
    bool    valid;
#endif
    
#ifdef SCENE_BACK_SURFACE
    float3  normal_back;        // world normal of the back surface of the model
#endif
    
    float3 ior; // Index of Refraction
    
    float roughness;
    float metallic;
    float reflectance;
    float ambientOcclusion; // default 1.0

#if defined(SHADING_MODEL_CLEAR_COAT)
    float   clearCoat;
    float   clearCoatRoughness;
    #if defined (MATERIAL_HAS_CLEAR_COAT_NORMAL)
    float3  clearCoatNormal;    // default float3(0.0, 0.0, 1.0)
    #endif
#endif

#if defined(SHADING_MODEL_IRIDESCENCE)
    float   thickness; // default to 300.0
#endif
    
#if defined(SHADING_MODEL_SUBSURFACE)
    float3    subsurfaceColor;    // default float3(1.0)
    float   subsurfacePower;    // default to 12.234
    float   subsurfaceThickness;// default to 1.0
#endif

};
#endif