#include "../../math/const.wgsl"
#include "../shadingData/shadingData.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using lambert equation
use:
    - <float> diffuseLambert(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness])
    - <float> diffuseLambert(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn diffuseLambertConstant() -> f32 { return INV_PI; }
fn diffuseLambertConstanta(shadingData: ShadingData) -> f32 { return diffuseLambertConstant(); }

fn diffuseLambert3(L: vec3f, N: vec3f) -> f32 { return max(0.0, dot(N, L)); }
fn diffuseLambert(shadingData: ShadingData) -> f32 { return max(0.0, shadingData.NoL); }
