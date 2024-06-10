/*
contributors: Patricio Gonzalez Vivo
description: bump in a range between -1 and 1
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn bump(x: f32, k: f32) -> f32 { return saturate( (1.0 - x * x) - k); }
fn bump2(x: vec2f, k: vec2f) -> vec2f { return saturate( (1.0 - x * x) - k); }
fn bump3(x: vec3f, k: vec3f) -> vec3f { return saturate( (1.0 - x * x) - k); }
fn bump4(x: vec4f, k: vec4f) -> vec4f { return saturate( (1.0 - x * x) - k); }
