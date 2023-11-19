#include "albedo.glsl"
#include "specular.glsl"
#include "emissive.glsl"
#include "occlusion.glsl"
#include "normal.glsl"
#include "metallic.glsl"
#include "roughness.glsl"
#include "shininess.glsl"

#include "../material.glsl"
// #include "../shadow.glsl"
#include "../ior.glsl"
#include "../../sample.glsl"

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
    - SHADING_MODEL_CLOTH
*/

#ifndef SURFACE_POSITION
#define SURFACE_POSITION vec3(0.0, 0.0, 0.0)
#endif

#ifndef FNC_MATERIAL_NEW
#define FNC_MATERIAL_NEW

void materialNew(out Material _mat) {
    // Surface data
    _mat.position           = (SURFACE_POSITION).xyz;
    _mat.normal             = materialNormal();

    #if defined(SCENE_BACK_SURFACE) && defined(RESOLUTION)
        vec4 back_surface       = SAMPLER_FNC(SCENE_BACK_SURFACE, gl_FragCoord.xy / RESOLUTION);
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

    _mat.ior                = vec3(IOR_GLASS_RGB);      // Index of Refraction
    _mat.f0                 = vec3(0.04, 0.04, 0.04);   // reflectance at 0 degree

    // Shade
    _mat.ambientOcclusion   = materialOcclusion();

    // _mat.shadow             = SHADOW_INIT;

    // Clear Coat Model
// #if defined(MATERIAL_HAS_CLEAR_COAT)
    _mat.clearCoat          = 0.0;
    _mat.clearCoatRoughness = 0.01;
#if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    _mat.clearCoatNormal    = vec3(0.0, 0.0, 1.0);
#endif
// #endif

    // SubSurface Model
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
        vec3 shadowCoord = LIGHT_COORD.xyz / LIGHT_COORD.w;
        float Di = SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy).r;
        float Do = LIGHT_COORD.z;
        float delta = Do - Di;

        #if defined(LIGHT_SHADOWMAP_SIZE) && !defined(PLATFORM_RPI)
        vec2 shadowmap_pixel = 1.0/vec2(LIGHT_SHADOWMAP_SIZE);
        shadowmap_pixel *= pow(delta, 0.6) * 20.0;

        Di = 0.0;
        for (float x= -2.0; x <= 2.0; x++)
            for (float y= -2.0; y <= 2.0; y++) 
                Di += SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy + vec2(x,y) * shadowmap_pixel).r;
        Di *= 0.04; // 1.0/25.0
        delta = Do - Di;
        #endif

        // This is pretty much of a hack by overwriting the absorption to the thinkness
        _mat.subsurfaceThickness = max(Do - Di, 0.005) * 30.0;
    }


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
