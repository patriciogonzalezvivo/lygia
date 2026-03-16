#include "../material.wgsl"

/*
contributors: Shadi El Hajj
description: Add materials a and b, property by property, store result in r
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

// #define MATERIAL_ADD

fn materialAdd(a: Material, b: Material, r: Material) {
    r.albedo = a.albedo + b.albedo;
    r.emissive = a.emissive + b.emissive;
    r.position = a.position + b.position;
    r.normal = a.normal + b.normal;

    r.normal_back = a.normal_back + b.normal_back;

    r.ior = a.ior + b.ior;
    r.roughness = a.roughness + b.roughness;
    r.metallic = a.metallic + b.metallic;
    r.reflectance = a.reflectance + b.reflectance;
    r.ambientOcclusion = a.ambientOcclusion + b.ambientOcclusion;

    r.clearCoat = a.clearCoat + b.clearCoat;
    r.clearCoatRoughness = a.clearCoatRoughness + b.clearCoatRoughness;
    r.clearCoatNormal = a.clearCoatNormal + b.clearCoatNormal;

    r.thickness  = a.thickness + b.thickness;

    r.subsurfaceColor = a.subsurfaceColor + b.subsurfaceColor;
    r.subsurfacePower = a.subsurfacePower + b.subsurfacePower;
    r.subsurfaceThickness = a.subsurfaceThickness + b.subsurfaceThickness;
}
