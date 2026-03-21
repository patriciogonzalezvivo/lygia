#include "../../math/pow2.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Index of refraction to reflectance at 0 degree https://handlespixels.wordpress.com/tag/f0-reflectance/
use: <float|vec3|vec4> ior2f0(<float|vec3|vec4> ior)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ior2f0(ior: f32) -> f32 { return pow2(ior - 1.0) / pow2(ior + 1.0); }
fn ior2f03(ior: vec3f) -> vec3f { return pow2(ior - 1.0) / pow2(ior + 1.0); }
fn ior2f04(ior: vec4f) -> vec4f { return vec4f(pow2(ior.rgb - 1.0) / pow2(ior.rgb + 1.0), ior.a); }
