#include "../common/schlick.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using burley equation
use:
    - <float> diffuseBurley(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness])
    - <float> diffuseBurley(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn diffuseBurley(NoV: f32, NoL: f32, LoH: f32, linearRoughness: f32) -> f32 {
    // Burley 2012, "Physically-Based Shading at Disney"
    let f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
    let lightScatter = schlick(1.0, f90, NoL);
    let viewScatter = schlick(1.0, f90, NoV);
    return lightScatter * viewScatter;
}

fn diffuseBurleya(shadingData: ShadingData) -> f32 {
    let LoH = dot(shadingData.L, shadingData.H);
    return diffuseBurley(shadingData.NoV, shadingData.NoL, LoH, shadingData.linearRoughness);
}
