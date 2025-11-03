/*
contributors: Patricio Gonzalez Vivo
description: power of 3
use: pow3(v: f32) -> f32 | pow32(v: vec2f) -> vec2f | pow33(v: vec3f) -> vec3f | pow34(v: vec4f) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn pow3(v: f32) -> f32 { return v * v * v; }
fn pow32(v: vec2f) -> vec2f { return v * v * v; }
fn pow33(v: vec3f) -> vec3f { return v * v * v; }
fn pow34(v: vec4f) -> vec4f { return v * v * v; }
