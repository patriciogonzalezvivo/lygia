#include "levels/inputRange.wgsl"
#include "levels/outputRange.wgsl"
#include "levels/gamma.wgsl"

/*
contributors: Johan Ismael
description: |
    Combines inputRange, outputRange and gamma functions into one
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levels(<vec3|vec4> color, <float|vec3> minInput, <float|vec3> gamma, <float|vec3 maxInput, <float|vec3 minOutput, <float|vec3 maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn levels3(v: vec3f, iMin: vec3f, g: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f {
    return levelsOutputRange( levelsGamma( levelsInputRange(v, iMin, iMax), g), oMin, oMax);
}

fn levels3a(v: vec3f, iMin: f32, g: f32, iMax: f32, oMin: f32, oMax: f32) -> vec3f {
    return levels(v, vec3f(iMin), vec3f(g), vec3f(iMax), vec3f(oMin), vec3f(oMax));
}

fn levels4(v: vec4f, iMin: vec3f, g: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec4f {
    return vec4f(levels(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}

fn levels4a(v: vec4f, iMin: f32, g: f32, iMax: f32, oMin: f32, oMax: f32) -> vec4f {
    return vec4f(levels(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}
