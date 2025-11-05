/*
contributors: Patricio Gonzalez Vivo
description: Convert from linear to gamma color space (assumes gamma 2.2).
use: linear2gamma(<f32|vec3f|vec4f> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER: make this a @param const when WESL supports that feature
const GAMMA: f32 = 2.2;
const INV_GAMMA: f32 = 1.0 / 2.2;

fn linear2gamma(v: f32) -> f32 {
    return pow(v, INV_GAMMA);
}

fn linear2gamma3(v: vec3f) -> vec3f {
    return pow(v, vec3f(INV_GAMMA));
}

fn linear2gamma4(v: vec4f) -> vec4f {
    return vec4f(linear2gamma3(v.rgb), v.a);
}
