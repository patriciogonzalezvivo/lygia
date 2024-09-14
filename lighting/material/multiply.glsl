#include "../material.glsl"

/*
contributors: Shadi El Hajj
description: Mutiply material properties by a constant, store result in r
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef MATERIAL_MULTIPLY
#define MATERIAL_MULTIPLY

void materialMultiply(Material mat, float f, Material r) {
    r.albedo = mat.albedo * f;
    r.emissive = mat.emissive * f;
    r.position = mat.position * f;
    r.normal = mat.normal * f;
    #if defined(SCENE_BACK_SURFACE)
    r.normal_back = mat.normal_back * f;
    #endif
    r.ior = mat.ior * f;
    r.roughness = mat.roughness * f;
    r.metallic = mat.metallic * f;
    r.reflectance = mat.reflectance * f;
    r.ambientOcclusion = mat.ambientOcclusion * f;
    #if defined(SHADING_MODEL_CLEAR_COAT)
    r.clearCoat = mat.clearCoat * f;
    r.clearCoatRoughness = mat.clearCoatRoughness * f;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    r.clearCoatNormal = mat.clearCoatNormal * f;
    #endif
    #endif
    #if defined(SHADING_MODEL_IRIDESCENCE)
    r.thickness = mat.thickness * f;
    #endif
    #if defined(SHADING_MODEL_SUBSURFACE)
    r.subsurfaceColor = mat.subsurfaceColor * f;
    r.subsurfacePower = mat.subsurfacePower * f;
    r.subsurfaceThickness = mat.subsurfaceThickness * f;
    #endif
}

#endif
