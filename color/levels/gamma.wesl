/*
contributors: Johan Ismael
description: |
    Color gamma correction similar to Levels adjustment in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsGamma(<vec3f|vec4f> color, <f32|vec3f> gamma)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn levelsGamma3(v: vec3f, g: vec3f) -> vec3f {
    return pow(v, 1.0 / g);
}

fn levelsGamma3Float(v: vec3f, g: f32) -> vec3f {
    return levelsGamma3(v, vec3f(g));
}

fn levelsGamma4(v: vec4f, g: vec3f) -> vec4f {
    return vec4f(levelsGamma3(v.rgb, g), v.a);
}

fn levelsGamma4Float(v: vec4f, g: f32) -> vec4f {
    return vec4f(levelsGamma3(v.rgb, vec3f(g)), v.a);
}
