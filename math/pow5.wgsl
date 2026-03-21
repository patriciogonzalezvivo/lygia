/*
contributors: Patricio Gonzalez Vivo
description: power of 5
use: <float|vec2|vec3|vec4> pow5(<float|vec2|vec3|vec4> v)
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
