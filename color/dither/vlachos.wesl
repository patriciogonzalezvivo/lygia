import lygia::math::decimate::decimate3;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Vlachos 2016, "Advanced VR Rendering"
    http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
use:
    - <vec4f|vec3f|f32> ditherVlachos(<vec4f|vec3f|f32> value, <vec2f> st, <i32> precision)
    - <vec4f|vec3f|f32> ditherVlachos(<vec4f|vec3f|f32> value, <vec2f> st)
    - <vec3f> ditherVlachos(<vec2f> st)
note: |
    The constant DITHER_VLACHOS_PRECISION can be customized (default: 256)
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER make this a @param const when WESL supports that
const DITHER_VLACHOS_PRECISION: i32 = 256;
// LATER support DITHER_VLACHOS_TIME 

// Core Vlachos noise function for scalar
fn ditherVlachos(b: f32, ist: vec2f) -> f32 {
    let st = ist;

    let noise = dot(vec2f(171.0, 231.0), st);
    let noise_fract = fract(noise / 71.0);
    let noise_normalized = (noise_fract * 2.0) - 1.0;
    return b + (noise_normalized / 255.0);
}

// Core Vlachos noise function returning vec3
fn ditherVlachos3Noise(ist: vec2f) -> vec3f {
    let st = ist;

    var noise = vec3f(dot(vec2f(171.0, 231.0), st));
    noise = fract(noise / vec3f(103.0, 71.0, 97.0));
    return noise;
}

// Main dithering function for vec3 with precision control
fn ditherVlachos3Precision(color: vec3f, st: vec2f, pres: i32) -> vec3f {
    let d = f32(pres);
    let ditherPattern = ditherVlachos3Noise(st);
    return decimate3(color + ditherPattern / d, vec3f(d));
}

// Overloads with xy coordinate using default precision
fn ditherVlachos3(color: vec3f, xy: vec2f) -> vec3f {
    return ditherVlachos3Precision(color, xy, DITHER_VLACHOS_PRECISION);
}

fn ditherVlachos4(color: vec4f, xy: vec2f) -> vec4f {
    return vec4f(ditherVlachos3Precision(color.rgb, xy, DITHER_VLACHOS_PRECISION), color.a);
}
