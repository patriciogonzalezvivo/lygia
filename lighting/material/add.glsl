#include "../material.glsl"

/*
contributors: Shadi El Hajj
description: Add materials a and b, property by property, store result in r
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef MATERIAL_ADD
#define MATERIAL_ADD

void materialAdd(Material a, Material b, Material r) {
    r.albedo = a.albedo + b.albedo;
    r.emissive = a.emissive + b.emissive;
    r.position = a.position + b.position;
    r.normal = a.normal + b.normal;

    #if defined(SCENE_BACK_SURFACE)
    r.normal_back = a.normal_back + b.normal_back;
    #endif

    r.ior = a.ior + b.ior;
    r.roughness = a.roughness + b.roughness;
    r.metallic = a.metallic + b.metallic;
    r.reflectance = a.reflectance + b.reflectance;
    r.ambientOcclusion = a.ambientOcclusion + b.ambientOcclusion;

    #if defined(SHADING_MODEL_CLEAR_COAT)
    r.clearCoat = a.clearCoat + b.clearCoat;
    r.clearCoatRoughness = a.clearCoatRoughness + b.clearCoatRoughness;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    r.clearCoatNormal = a.clearCoatNormal + b.clearCoatNormal;
    #endif
    #endif

    #if defined(SHADING_MODEL_IRIDESCENCE)
    r.thickness  = a.thickness + b.thickness;
    #endif

    #if defined(SHADING_MODEL_SUBSURFACE)
    r.subsurfaceColor = a.subsurfaceColor + b.subsurfaceColor;
    r.subsurfacePower = a.subsurfacePower + b.subsurfacePower;
    r.subsurfaceThickness = a.subsurfaceThickness + b.subsurfaceThickness;
    #endif
}

#endif
