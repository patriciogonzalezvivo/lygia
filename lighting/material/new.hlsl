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
#include "../../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Material Constructor.
use:
    - void materialNew(out <material> _mat)
    - <material> materialNew()
options:
    - SURFACE_POSITION
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

#ifndef SURFACE_POSITION
#define SURFACE_POSITION float3(0.0, 0.0, 0.0)
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 20.0
#endif


#ifndef FNC_MATERIAL_NEW
#define FNC_MATERIAL_NEW

void materialNew(out Material _mat) {
    // Surface data
    _mat.position           = (SURFACE_POSITION).xyz;
    _mat.normal             = materialNormal();

#if defined(RENDER_RAYMARCHING)
    _mat.sdf                = RAYMARCH_MAX_DIST;
    _mat.valid              = true;
#endif
    
    #if defined(SCENE_BACK_SURFACE) && defined(RESOLUTION)
        float4 back_surface       = SAMPLER_FNC(SCENE_BACK_SURFACE, gl_FragCoord.xy / RESOLUTION);
        _mat.normal_back        = back_surface.xyz;
    #else
        #if defined(SCENE_BACK_SURFACE)
        // Naive assumption of the back surface
        _mat.normal_back        = -_mat.normal;
        #endif
    #endif

    // PBR Properties
    _mat.albedo             = materialAlbedo();
    _mat.emissive           = materialEmissive();
    _mat.roughness          = materialRoughness();
    _mat.metallic           = materialMetallic();
    _mat.reflectance        = 0.5;

    _mat.ior                = float3(IOR_GLASS_RGB);      // Index of Refraction

    _mat.ambientOcclusion   = materialOcclusion();

#if defined(SHADING_MODEL_CLEAR_COAT)
    _mat.clearCoat          = 0.0;
    _mat.clearCoatRoughness = 0.01;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    _mat.clearCoatNormal    = float3(0.0, 0.0, 1.0);
    #endif
#endif

#if defined(SHADING_MODEL_IRIDESCENCE)
    _mat.thickness          = 300.0;
#endif
    
#if defined(SHADING_MODEL_SUBSURFACE)
    _mat.subsurfaceColor    = _mat.albedo.rgb;
    _mat.subsurfacePower    = 12.234;
    _mat.subsurfaceThickness = 20.0;

    // Simulate Absorption Using Depth Map (shadowmap)
    // https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-16-real-time-approximations-subsurface-scattering
    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_COORD)
    {
        float3 shadowCoord = LIGHT_COORD.xyz / LIGHT_COORD.w;
        float Di = SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy).r;
        float Do = LIGHT_COORD.z;
        float delta = Do - Di;

        #if defined(LIGHT_SHADOWMAP_SIZE) && !defined(PLATFORM_RPI)
        float2 shadowmap_pixel = 1.0/float2(LIGHT_SHADOWMAP_SIZE, LIGHT_SHADOWMAP_SIZE);
        shadowmap_pixel *= pow(delta, 0.6) * 20.0;

        Di = 0.0;
        for (float x= -2.0; x <= 2.0; x++)
            for (float y= -2.0; y <= 2.0; y++) 
                Di += SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy + float2(x,y) * shadowmap_pixel).r;
        Di *= 0.04; // 1.0/25.0
        delta = Do - Di;
        #endif

        // This is pretty much of a hack by overwriting the absorption to the thinkness
        _mat.subsurfaceThickness = max(Do - Di, 0.005) * 30.0;
    }
    #endif

#endif

}

Material materialNew() {
    Material mat;
    materialNew(mat);
    return mat;
}

Material materialNew(float3 albedo, float sdf) {
    Material mat = materialNew();
    mat.albedo.rgb = albedo;
    mat.sdf = sdf;
    return mat;
}

Material materialNew(float3 albedo, float roughness, float metallic, float sdf) {
    Material mat = materialNew();
    mat.albedo.rgb = albedo;
    mat.metallic = metallic;
    mat.roughness = roughness;
    mat.sdf = sdf;
    return mat;
}

#endif
