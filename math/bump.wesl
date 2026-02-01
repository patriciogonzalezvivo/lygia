/*
contributors: Patricio Gonzalez Vivo
description: bump in a range between -1 and 1
use: <f32|vec3f> bump(<f32|vec3f> x [, <f32|vec3f> k])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::math::saturate::{saturate, saturate2, saturate3, saturate4};

// 2-arg versions (with threshold k)
fn bump(x: f32, k: f32) -> f32 { return saturate((1.0 - x * x) - k); }
fn bump2(x: vec2f, k: vec2f) -> vec2f { return saturate2((1.0 - x * x) - k); }
fn bump3(x: vec3f, k: vec3f) -> vec3f { return saturate3((1.0 - x * x) - k); }
fn bump4(x: vec4f, k: vec4f) -> vec4f { return saturate4((1.0 - x * x) - k); }

// 1-arg versions (default k=0, using max for exact GLSL match)
fn bumpDefault(x: f32) -> f32 { return max(1.0 - x * x, 0.0); }
fn bumpDefault2(x: vec2f) -> vec2f { return max(vec2f(1.0) - x * x, vec2f(0.0)); }
fn bumpDefault3(x: vec3f) -> vec3f { return max(vec3f(1.0) - x * x, vec3f(0.0)); }
fn bumpDefault4(x: vec4f) -> vec4f { return max(vec4f(1.0) - x * x, vec4f(0.0)); }
