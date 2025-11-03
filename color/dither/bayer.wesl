import lygia::math::decimate::decimate;
import lygia::math::decimate::decimate3;

/*
contributors: Patricio Gonzalez Vivo
description: Dither using a 8x8 Bayer matrix
use:
    - <vec4f|vec3f|f32> ditherBayer(<vec4f|vec3f|f32> value, <vec2f> st, <i32> precision)
    - <vec4f|vec3f|f32> ditherBayer(<vec4f|vec3f|f32> value, <vec2f> st)
    - <vec4f|vec3f|f32> ditherBayer(<vec4f|vec3f|f32> value, <i32> precision)
    - <vec4f|vec3f|f32> ditherBayer(<vec4f|vec3f|f32> value)
    - <f32> ditherBayer(<vec2f> xy)
note: |
    The constant DITHER_BAYER_PRECISION can be customized (default: 256)
examples:
    - /shaders/color_dither_bayer.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER make this a @param const when WESL supports that
const DITHER_BAYER_PRECISION: i32 = 256;

// Core Bayer matrix function using 8x8 pattern
fn ditherBayer(xy: vec2f) -> f32 {
    let x = xy.x % 8.0;
    let y = xy.y % 8.0;

    // 8x8 Bayer matrix values / 64.0
    let bayerMatrix = array<f32, 64>(
        0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
       48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
       12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, 38.0/64.0,
       60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
        3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
       51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
       15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, 37.0/64.0,
       63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
    );

    let index = i32(x) + i32(y) * 8;
    return bayerMatrix[index];
}

// Main dithering function for vec3 with precision control
fn ditherBayer3Precision(color: vec3f, xy: vec2f, pres: i32) -> vec3f {
    let d = vec3f(f32(pres));
    let decimated = decimate3(color, d);
    let diff = (color - decimated) * d;
    let ditherPattern = vec3f(ditherBayer(xy));
    return decimate3(color + (step(ditherPattern, diff) / d), d);
}

// Overloads with xy coordinate and optional precision
fn ditherBayerPrecision(val: f32, xy: vec2f, pres: i32) -> f32 {
    return ditherBayer3Precision(vec3f(val), xy, pres).r;
}

fn ditherBayer3(color: vec3f, xy: vec2f) -> vec3f {
    return ditherBayer3Precision(color, xy, DITHER_BAYER_PRECISION);
}

fn ditherBayer4(color: vec4f, xy: vec2f) -> vec4f {
    return vec4f(ditherBayer3Precision(color.rgb, xy, DITHER_BAYER_PRECISION), color.a);
}