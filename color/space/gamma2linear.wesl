/*
contributors: Patricio Gonzalez Vivo
description: Convert from gamma to linear color space (assumes gamma 2.2).
use: gamma2linear(<f32|vec3f|vec4f> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER: make this a @param const when WESL supports that feature
const GAMMA: f32 = 2.2;

fn gamma2linear(v: f32) -> f32 {
    return pow(v, GAMMA);
}

fn gamma2linear3(v: vec3f) -> vec3f {
    return pow(v, vec3f(GAMMA));
}

fn gamma2linear4(v: vec4f) -> vec4f {
    return vec4f(gamma2linear3(v.rgb), v.a);
}
