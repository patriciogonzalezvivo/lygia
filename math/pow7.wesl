/*
contributors: Patricio Gonzalez Vivo
description: power of 7
use: pow7(v: f32) -> f32 | pow72(v: vec2f) -> vec2f | pow73(v: vec3f) -> vec3f | pow74(v: vec4f) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn pow7(v: f32) -> f32 { return v * v * v * v * v * v * v; }
fn pow72(v: vec2f) -> vec2f { return v * v * v * v * v * v * v; }
fn pow73(v: vec3f) -> vec3f { return v * v * v * v * v * v * v; }
fn pow74(v: vec4f) -> vec4f { return v * v * v * v * v * v * v; }
