/*
contributors: Patricio Gonzalez Vivo
description: Flips the float passed in, 0 becomes 1 and 1 becomes 0
use: flip(<float> v, <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn flip(v: f32, pct: f32) -> f32 {
    return mix(v, 1.0 - v, pct);
}

fn flip3(v: vec3f, pct: f32) -> vec3f {
    return mix(v, 1.0 - v, pct);
}

fn flip4(v: vec4f, pct: f32) -> vec4f {
    return mix(v, 1.0 - v, pct);
}
