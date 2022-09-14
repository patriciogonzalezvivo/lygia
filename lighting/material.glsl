#include "material/baseColor.glsl"
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
    vec4    baseColor;
    vec3    emissive;

    vec3    position;       // world position of the surface
    vec3    normal;         // world normal of the surface
    
    vec3    f0;             // = vec3(0.04);
    float   reflectance;    // = 0.5;

    float   roughness;
    float   metallic;
    float   ambientOcclusion;
    float   shadow;         // = 1.0;

#if defined(MATERIAL_CLEARCOAT_THICKNESS)
    float   clearCoat;
    float   clearCoatRoughness;
    #if defined(MATERIAL_CLEARCOAT_THICKNESS_NORMAL)
    vec3    clearCoatNormal;// = vec3(0.0, 0.0, 1.0);
    #endif
#endif

#if defined(SHADING_MODEL_SUBSURFACE)
    float   thickness;      // = 0.5;
    float   subsurfacePower;// = 12.234;
#endif

#if defined(SHADING_MODEL_CLOTH)
    vec3    sheenColor;
#endif

#if defined(MATERIAL_SUBSURFACE_COLOR)
    vec3    subsurfaceColor;// = vec3(1.0);
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    vec3    specularColor;
    float   glossiness;
#endif

};
#endif