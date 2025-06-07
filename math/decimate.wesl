/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific precision
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn decimate(v: f32, d:f32) -> f32 { return floor(v * d) / d; }
fn decimate2(v: vec2f, d:vec2f) -> vec2f { return floor(v * d) / d; }
fn decimate3(v: vec3f, d:vec3f) -> vec3f { return floor(v * d) / d; }
fn decimate4(v: vec4f, d:vec4f) -> vec4f { return floor(v * d) / d; }