#include "../../math/decimate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Dither using a 8x8 Bayer matrix
use:
    - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st)
    - <float> ditherBayer(<vec2> xy)
options:
    - DITHER_BAKER_COORD
examples:
    - /shaders/color_dither_bayer.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mod_fract(a: vec2f, b: f32) -> vec2f { return modf(a / b).fract * b; }

const indexMatrix4x4 = array<f32, 16>(0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);

fn indexValue4(fragCoord: vec2<f32>) -> f32 {
    let p = vec2<i32>(mod_fract(fragCoord, 4));
    return indexMatrix4x4[(p.x + p.y * 4)] / 16.0;
}

const indexMatrix8x8 =
    array<f32, 64>(0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28,
    52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39,
    13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21);

fn indexValue8(fragCoord: vec2<f32>) -> f32 {
    let p = vec2<i32>(mod_fract(fragCoord, 8));
    return indexMatrix8x8[(p.x + p.y * 8)] / 64.0;
}

fn ditherBayer1(ist: vec2f) -> f32 { return indexValue4(ist); }

fn ditherBayer3(color: vec3f, xy: vec2f, d: vec3f) -> vec3f {
    let decimated = decimate3(color, d);
    let diff = (color - decimated) * d;
    let ditherPattern = vec3(ditherBayer1(xy));
    return decimate3(color + (step(ditherPattern, diff) / d), d);
}