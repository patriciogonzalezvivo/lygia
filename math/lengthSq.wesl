/*
contributors: Patricio Gonzalez Vivo
description: Squared length
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lengthSq2(v: vec2f) -> f32 { return dot(v, v); }
fn lengthSq3(v: vec3f) -> f32 { return dot(v, v); }
fn lengthSq4(v: vec4f) -> f32 { return dot(v, v); }
