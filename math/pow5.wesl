/*
contributors: Patricio Gonzalez Vivo
description: power of 5
use: pow5(v: f32) -> f32 | pow52(v: vec2f) -> vec2f | pow53(v: vec3f) -> vec3f | pow54(v: vec4f) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn pow5(v: f32) -> f32 {
    let v2 = v * v;
    return v2 * v2 * v;
}

fn pow52(v: vec2f) -> vec2f {
    let v2 = v * v;
    return v2 * v2 * v;
}

fn pow53(v: vec3f) -> vec3f {
    let v2 = v * v;
    return v2 * v2 * v;
}

fn pow54(v: vec4f) -> vec4f {
    let v2 = v * v;
    return v2 * v2 * v;
}
