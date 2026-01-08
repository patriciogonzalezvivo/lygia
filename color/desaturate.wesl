/*
contributors: Patricio Gonzalez Vivo
description: Change saturation of a color
use: desaturate(<vec3f> color, <f32> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn desaturate(color: vec3f, amount: f32) -> vec3f {
    return mix(color, vec3f(dot(vec3f(0.3, 0.59, 0.11), color)), amount);
}

fn desaturate4(color: vec4f, amount: f32) -> vec4f {
    return vec4f(desaturate(color.rgb, amount), color.a);
}