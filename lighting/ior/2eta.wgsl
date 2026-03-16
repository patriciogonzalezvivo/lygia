/*
contributors: Patricio Gonzalez Vivo
description: Index of refraction to ratio of index of refraction
use: <float|vec3|vec4> ior2eta(<float|vec3|vec4> ior)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ior2eta(ior: f32) -> f32 { return 1.0/ior; }
fn ior2eta3(ior: vec3f) -> vec3f { return 1.0/ior; }
fn ior2eta4(ior: vec4f) -> vec4f { return vec4f(1.0/ior.rgb, ior.a); }
