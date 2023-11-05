#include "albedo.hlsl"
#include "specular.hlsl"
#include "emissive.hlsl"
#include "occlusion.hlsl"
#include "normal.hlsl"
#include "metallic.hlsl"
#include "roughness.hlsl"
#include "shininess.hlsl"

#include "../material.hlsl"
#include "../ior.hlsl"
#include "../../sample.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: 
    - void materialNew(out <material> _mat)
    - <material> materialNew()
options:
    - SURFACE_POSITION
    - SHADING_SHADOWS
    - MATERIAL_HAS_CLEAR_COAT
    - MATERIAL_CLEARCOAT_ROUGHNESS
    - MATERIAL_HAS_CLEAR_COAT_NORMAL
    - SHADING_MODEL_SUBSURFACE
    - MATERIAL_SUBSURFACE_COLOR
    - SHADING_MODEL_CLOTH
*/

#ifndef SURFACE_POSITION
#define SURFACE_POSITION float3(0.0, 0.0, 0.0)
#endif

#ifndef SHADOW_INIT
#if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
#define SHADOW_INIT shadow(LIGHT_SHADOWMAP, float2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z)
#else
#define SHADOW_INIT 1.0
#endif
#endif


#ifndef FNC_MATERIAL_NEW
#define FNC_MATERIAL_NEW

void materialNew(out Material _mat) {
    // Surface data
    _mat.position           = (SURFACE_POSITION).xyz;
    _mat.normal             = materialNormal();

    #if defined(SCENE_BACK_SURFACE) && defined(RESOLUTION)
    float4 back_surface       = SAMPLER_FNC(SCENE_BACK_SURFACE, gl_FragCoord.xy / RESOLUTION);
    _mat.normal_back        = back_surface.xyz;
    #if defined(SHADING_MODEL_SUBSURFACE)
    _mat.thickness          = saturate(gl_FragCoord.z - back_surface.a);
    #endif
    #else 
    #if defined(SCENE_BACK_SURFACE)
    _mat.normal_back        = -_mat.normal;
    #endif
    #if defined(SHADING_MODEL_SUBSURFACE)
    _mat.thickness          = 0.5;
    #endif
    #endif

    // PBR Properties
    _mat.albedo             = materialAlbedo();
    _mat.emissive           = materialEmissive();
    _mat.roughness          = materialRoughness();
    _mat.metallic           = materialMetallic();

    _mat.ior                = float3(IOR_GLASS_RGB);      // Index of Refraction
    _mat.f0                 = float3(0.04, 0.04, 0.04); // reflectance at 0 degree

    // Shade
    _mat.ambientOcclusion   = materialOcclusion();

    _mat.shadow             = SHADOW_INIT;

    // Clear Coat Model
    _mat.clearCoat          = 0.0;
    _mat.clearCoatRoughness = 0.01;
#if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    _mat.clearCoatNormal    = float3(0.0, 0.0, 1.0);
#endif

    // SubSurface Model
#if defined(SHADING_MODEL_SUBSURFACE)
    _mat.subsurfacePower    = 12.234;
#endif

#if defined(MATERIAL_SUBSURFACE_COLOR)
    #if defined(SHADING_MODEL_SUBSURFACE)
    _mat.subsurfaceColor    = float3(1.0, 1.0, 1.0);
    #else
    _mat.subsurfaceColor    = float3(0.0, 0.0, 0.0);
    #endif
#endif

    // Cloath Model
#if defined(SHADING_MODEL_CLOTH)
    _mat.sheenColor         = sqrt(_mat.albedo.rgb);
#endif
}

Material materialNew() {
    Material mat;
    materialNew(mat);
    return mat;
}

#endif
