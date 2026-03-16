/*
contributors: Patricio Gonzalez Vivo
description: Change saturation of a color
use: desaturate(<float|vec3|vec4> color, float amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn desaturate3(v: vec3f, a: f32) -> vec3f {
    return mix(v, vec3f(dot(vec3f(.3, .59, .11), v)), a);
}

fn desaturate4(v: vec4f, a: f32) -> vec4f {
    return vec4f(desaturate(v.rgb, a), v.a);
}
