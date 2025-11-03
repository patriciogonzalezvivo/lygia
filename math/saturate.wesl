/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and 1
use: saturate(v: f32) -> f32 | saturate2(v: vec2f) -> vec2f | saturate3(v: vec3f) -> vec3f | saturate4(v: vec4f) -> vec4f
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn saturate(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate2(v: vec2f) -> vec2f { return clamp(v, vec2(0.0), vec2(1.0)); }
fn saturate3(v: vec3f) -> vec3f { return clamp(v, vec3(0.0), vec3(1.0)); }
fn saturate4(v: vec4f) -> vec4f { return clamp(v, vec4(0.0), vec4(1.0)); }
