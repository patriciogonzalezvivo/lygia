#include "material/albedo.glsl"
#include "material/specular.glsl"
#include "material/emissive.glsl"
#include "material/occlusion.glsl"

#include "material/normal.glsl"

#include "material/metallic.glsl"
#include "material/roughness.glsl"

#include "material/shininess.glsl"

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
struct Material {
    vec4    albedo;
    vec3    emissive;

    vec3    position;       // world position of the surface
    vec3    normal;         // world normal of the surface

#if defined(RENDER_RAYMARCHING)
    float   sdf;
    bool    valid;
#endif

    #if defined(SCENE_BACK_SURFACE)
    vec3    normal_back;    // world normal of the back surface of the model
    #endif
    
    vec3    ior;            // Index of Refraction

    float   roughness;
    float   metallic;
    float   reflectance;
    float   ambientOcclusion;   // default 1.0

#if defined(SHADING_MODEL_CLEAR_COAT)
    float   clearCoat;
    float   clearCoatRoughness;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    vec3    clearCoatNormal;    // default vec3(0.0, 0.0, 1.0);
    #endif
#endif

#if defined(SHADING_MODEL_IRIDESCENCE)
    float   thickness; // default to 300.0
#endif

#if defined(SHADING_MODEL_SUBSURFACE)
    vec3    subsurfaceColor;    // default vec3(1.0)
    float   subsurfacePower;    // default to 12.234
    float   subsurfaceThickness;// default to 1.0
#endif

};
#endif
