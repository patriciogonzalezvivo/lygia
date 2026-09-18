#include "albedo.wgsl"
#include "specular.wgsl"
#include "emissive.wgsl"
#include "occlusion.wgsl"
#include "normal.wgsl"
#include "metallic.wgsl"
#include "roughness.wgsl"
#include "shininess.wgsl"

#include "../material.wgsl"
#include "../ior.wgsl"
#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
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

// #define SURFACE_POSITION vec3(0.0, 0.0, 0.0)

fn materialNew(_mat: Material) {
    const RAYMARCH_MAX_DIST: f32 = 20.0;
    // Surface data
    _mat.position           = (SURFACE_POSITION).xyz;
    _mat.normal             = materialNormal();

    _mat.sdf                = RAYMARCH_MAX_DIST;
    _mat.valid              = true;

        let back_surface = SAMPLER_FNC(SCENE_BACK_SURFACE, gl_FragCoord.xy / RESOLUTION);
        _mat.normal_back        = back_surface.xyz;
        // Naive assumption of the back surface
        _mat.normal_back        = -_mat.normal;

    // PBR Properties
    _mat.albedo             = materialAlbedo();
    _mat.emissive           = materialEmissive();
    _mat.roughness          = materialRoughness();
    _mat.metallic           = materialMetallic();
    _mat.reflectance        = 0.5;

    _mat.ior                = vec3f(IOR_GLASS_RGB);      // Index of Refraction

    _mat.ambientOcclusion   = materialOcclusion();

    _mat.clearCoat          = 0.0;
    _mat.clearCoatRoughness = 0.01;
    _mat.clearCoatNormal    = vec3f(0.0, 0.0, 1.0);

    _mat.thickness          = 300.0;

    _mat.subsurfaceColor    = _mat.albedo.rgb;
    _mat.subsurfacePower    = 12.234;
    _mat.subsurfaceThickness = 20.0;

    // Simulate Absorption Using Depth Map (shadowmap)
    // https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-16-real-time-approximations-subsurface-scattering
    {
        let shadowCoord = LIGHT_COORD.xyz / LIGHT_COORD.w;
        let Di = SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy).r;
        let Do = LIGHT_COORD.z;
        let delta = Do - Di;

        let shadowmap_pixel = 1.0/vec2f(LIGHT_SHADOWMAP_SIZE);
        shadowmap_pixel *= pow(delta, 0.6) * 20.0;

        Di = 0.0;
        for (float x= -2.0; x <= 2.0; x++)
            for (float y= -2.0; y <= 2.0; y++) 
                Di += SAMPLER_FNC(LIGHT_SHADOWMAP, LIGHT_COORD.xy + vec2f(x,y) * shadowmap_pixel).r;
        Di *= 0.04; // 1.0/25.0
        delta = Do - Di;

        // This is pretty much of a hack by overwriting the absorption to the thinkness
        _mat.subsurfaceThickness = max(Do - Di, 0.005) * 30.0;
    }

}

Material materialNew() {
    Material mat;
    materialNew(mat);
    return mat;
}

Material materialNew(vec3 albedo, float sdf) {
    Material mat = materialNew();
    mat.albedo.rgb = albedo;
    mat.sdf = sdf;
    return mat;
}

Material materialNew(vec3 albedo, float roughness, float metallic, float sdf) {
    Material mat = materialNew();
    mat.albedo.rgb = albedo;
    mat.metallic = metallic;
    mat.roughness = roughness;
    mat.sdf = sdf;
    return mat;
}
