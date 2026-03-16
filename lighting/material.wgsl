#include "material/albedo.wgsl"
#include "material/specular.wgsl"
#include "material/emissive.wgsl"
#include "material/occlusion.wgsl"

#include "material/normal.wgsl"

#include "material/metallic.wgsl"
#include "material/roughness.wgsl"

#include "material/shininess.wgsl"

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

// #define RENDER_RAYMARCHING
// #define SHADING_MODEL_CLEAR_COAT

struct Material {
    var albedo: vec4f;
    var emissive: vec3f;

    vec3    position;       // world position of the surface
    vec3    normal;         // world normal of the surface

    var sdf: f32;
    var valid: bool;

    vec3    normal_back;    // world normal of the back surface of the model
    
    vec3    ior;            // Index of Refraction

    var roughness: f32;
    var metallic: f32;
    var reflectance: f32;
    float   ambientOcclusion;   // default 1.0

    var clearCoat: f32;
    var clearCoatRoughness: f32;
    vec3    clearCoatNormal;    // default vec3f(0.0, 0.0, 1.0);

    float   thickness; // default to 300.0

    vec3    subsurfaceColor;    // default vec3f(1.0)
    float   subsurfacePower;    // default to 12.234
    float   subsurfaceThickness;// default to 1.0

};
