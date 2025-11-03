/*
contributors: Patricio Gonzalez Vivo
description: Bias high pass
use: contrast(<vec4f|vec3f|f32> value, <f32> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn contrast(v: f32, a: f32) -> f32 {
    return (v - 0.5) * a + 0.5;
}

fn contrast3(v: vec3f, a: f32) -> vec3f {
    return (v - 0.5) * a + 0.5;
}

fn contrast4(v: vec4f, a: f32) -> vec4f {
    return vec4f((v.rgb - 0.5) * a + 0.5, v.a);
}
