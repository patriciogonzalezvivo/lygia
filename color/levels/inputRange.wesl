/*
contributors: Johan Ismael
description: |
    Color input range adjustment similar to Levels adjustment tool in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsInputRange(<vec3f|vec4f> color, <f32|vec3f> minInput, <f32|vec3f> maxInput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn levelsInputRange3(v: vec3f, iMin: vec3f, iMax: vec3f) -> vec3f {
    return min(max(v - iMin, vec3f(0.0)) / (iMax - iMin), vec3f(1.0));
}

fn levelsInputRange3Float(v: vec3f, iMin: f32, iMax: f32) -> vec3f {
    return levelsInputRange3(v, vec3f(iMin), vec3f(iMax));
}

fn levelsInputRange4(v: vec4f, iMin: vec3f, iMax: vec3f) -> vec4f {
    return vec4f(levelsInputRange3(v.rgb, iMin, iMax), v.a);
}

fn levelsInputRange4Float(v: vec4f, iMin: f32, iMax: f32) -> vec4f {
    return vec4f(levelsInputRange3Float(v.rgb, iMin, iMax), v.a);
}
