/*
contributors: Patricio Gonzalez Vivo
description: Change the exposure of a color
use: exposure(<f32|vec3f|vec4f> color, <f32> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn exposure(value: f32, amount: f32) -> f32 {
    return value * pow(2.0, amount);
}

fn exposure3(color: vec3f, amount: f32) -> vec3f {
    return color * pow(2.0, amount);
}

fn exposure4(color: vec4f, amount: f32) -> vec4f {
    return vec4f(exposure3(color.rgb, amount), color.a);
}
