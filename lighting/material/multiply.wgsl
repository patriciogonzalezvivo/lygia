#include "../material.wgsl"

/*
contributors: Shadi El Hajj
description: Multiply material properties by a constant, store result in r
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

// #define MATERIAL_MULTIPLY

fn materialMultiply(mat: Material, f: f32, r: Material) {
    r.albedo = mat.albedo * f;
    r.emissive = mat.emissive * f;
    r.position = mat.position * f;
    r.normal = mat.normal * f;
    r.normal_back = mat.normal_back * f;
    r.ior = mat.ior * f;
    r.roughness = mat.roughness * f;
    r.metallic = mat.metallic * f;
    r.reflectance = mat.reflectance * f;
    r.ambientOcclusion = mat.ambientOcclusion * f;
    r.clearCoat = mat.clearCoat * f;
    r.clearCoatRoughness = mat.clearCoatRoughness * f;
    r.clearCoatNormal = mat.clearCoatNormal * f;
    r.thickness = mat.thickness * f;
    r.subsurfaceColor = mat.subsurfaceColor * f;
    r.subsurfacePower = mat.subsurfacePower * f;
    r.subsurfaceThickness = mat.subsurfaceThickness * f;
}
