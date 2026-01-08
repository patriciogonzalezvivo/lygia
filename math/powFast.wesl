/*
contributors: Patricio Gonzalez Vivo
description: fast approximation to pow()
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn powFast(a: f32, b: f32) -> f32 { return a / ((1.0 - b) * a + b); }
fn powFast2(a: vec2f, b: vec2f) -> vec2f { return a / ((1.0 - b) * a + b); }
fn powFast3(a: vec3f, b: vec3f) -> vec3f { return a / ((1.0 - b) * a + b); }
fn powFast4(a: vec4f, b: vec4f) -> vec4f { return a / ((1.0 - b) * a + b); }
