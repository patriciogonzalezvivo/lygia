#include "../../math/decimate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Jimenez 2014, "Next Generation Post-Processing Call of Duty" http://advances.realtimerendering.com/s2014/index.html'
use: <vec4|vec3|float> interleavedGradientNoise(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_INTERLEAVEDGRADIENTNOISE_TIME
    - DITHER_INTERLEAVEDGRADIENTNOISE_COORD
    - DITHER_INTERLEAVEDGRADIENTNOISE_CHROMATIC
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define DITHER_INTERLEAVEDGRADIENTNOISE_COORD gl_FragCoord.xy

// #define DITHER_INTERLEAVEDGRADIENTNOISE_TIME DITHER_TIME

// #define DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION DITHER_PRECISION

fn ditherInterleavedGradientNoise2(st: vec2f) -> f32 {
    st += 1337.0*fract(DITHER_INTERLEAVEDGRADIENTNOISE_TIME);
    st = floor(st);
    return fract(52.982919 * fract(dot(vec2f(0.06711, 0.00584), st))) * 2.0 - 1.0;
}

fn ditherInterleavedGradientNoise(value: f32, st: vec2f, pres: i32) -> f32 {
    let ditherPattern = ditherInterleavedGradientNoise(st);
    return value + ditherPattern / 255.0;
}

fn ditherInterleavedGradientNoise3(color: vec3f, st: vec2f, pres: i32) -> vec3f {
    vec3 ditherPattern = vec3f(
            ditherInterleavedGradientNoise(st),
            ditherInterleavedGradientNoise(st + 0.1337),
            ditherInterleavedGradientNoise(st + 0.3141));
    let ditherPattern = vec3f(ditherInterleavedGradientNoise(st));
    
    // return color + ditherPattern / 255.0;

    let d = float(pres);
    let h = 0.5 / d;
    // vec3 decimated = decimate(color, d);
    // vec3 diff = (color - decimated) * d;
    // ditherPattern = step(ditherPattern, diff);
    return decimate(color - h + ditherPattern / d, d);
}

// float ditherInterleavedGradientNoise(const float b, const vec2 st) { return b + triangleNoise(st) / float(DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
fn ditherInterleavedGradientNoise3a(color: vec3f, xy: vec2f) -> vec3f {  return ditherInterleavedGradientNoise(color, xy, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
fn ditherInterleavedGradientNoise4(color: vec4f, xy: vec2f) -> vec4f {  return vec4f(ditherInterleavedGradientNoise(color.rgb, xy, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION), color.a); }

fn ditherInterleavedGradientNoisea(val: f32, pres: i32) -> f32 { return ditherInterleavedGradientNoise(vec3f(val),DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres).r; }
fn ditherInterleavedGradientNoise3b(color: vec3f, pres: i32) -> vec3f { return ditherInterleavedGradientNoise(color, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres); }
fn ditherInterleavedGradientNoise4a(color: vec4f, pres: i32) -> vec4f { return vec4f(ditherInterleavedGradientNoise(color.rgb, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres), color.a); }

fn ditherInterleavedGradientNoiseb(val: f32) -> f32 { return ditherInterleavedGradientNoise(vec3f(val), DITHER_INTERLEAVEDGRADIENTNOISE_COORD, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION).r; }
fn ditherInterleavedGradientNoise3c(color: vec3f) -> vec3f { return ditherInterleavedGradientNoise(color, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
fn ditherInterleavedGradientNoise4b(color: vec4f) -> vec4f { return vec4f(ditherInterleavedGradientNoise(color.rgb), color.a); }
