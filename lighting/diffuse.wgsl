#include "shadingData/new.wgsl"
#include "diffuse/lambert.wgsl"
#include "diffuse/orenNayar.wgsl"
#include "diffuse/burley.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution
use: lightSpot(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define DIFFUSE_FNC diffuseLambert
// #define DIFFUSE_FNC diffuseOrenNayar

fn diffuse(shadingData: ShadingData) -> f32 { return DIFFUSE_FNC(shadingData); }
