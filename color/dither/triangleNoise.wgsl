#include "../../math/decimate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: "2016, Banding in Games: A Noisy Rant"
use:
    - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <float> time)
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define DITHER_TRIANGLENOISE_COORD gl_FragCoord.xy

// #define DITHER_TRIANGLENOISE_TIME DITHER_TIME

// #define DITHER_TRIANGLENOISE_CHROMATIC

// #define DITHER_TRIANGLENOISE_PRECISION DITHER_PRECISION
const DITHER_TRIANGLENOISE_PRECISION: f32 = 255;

fn triangleNoise(st: HIGHP) -> f32 {
    st = floor(st);
    st += vec2f(0.07 * fract(DITHER_TRIANGLENOISE_TIME));
    st  = fract(st * vec2f(5.3987, 5.4421));
    st += dot(st.yx, st.xy + vec2f(21.5351, 14.3137));

    HIGHP float xy = st.x * st.y;
    return (fract(xy * 95.4307) + fract(xy * 75.04961) - 1.0);
}

fn ditherTriangleNoise3(color: vec3f, st: HIGHP, pres: i32) -> vec3f {
    
    vec3 ditherPattern = vec3f(
            triangleNoise(st),
            triangleNoise(st + 0.1337),
            triangleNoise(st + 0.3141));
    let ditherPattern = vec3f(triangleNoise(st));
    
    // return color + ditherPattern / 255.0;
    let d = float(pres);
    let h = 0.5/d;
    return decimate(color - h + ditherPattern / d, d);
}

fn ditherTriangleNoise(b: f32, st: HIGHP) -> f32 { return b + triangleNoise(st) / float(DITHER_TRIANGLENOISE_PRECISION); }
fn ditherTriangleNoise3a(color: vec3f, xy: vec2f) -> vec3f {  return ditherTriangleNoise(color, xy, DITHER_TRIANGLENOISE_PRECISION); }
fn ditherTriangleNoise4(color: vec4f, xy: vec2f) -> vec4f {  return vec4f(ditherTriangleNoise(color.rgb, xy, DITHER_TRIANGLENOISE_PRECISION), color.a); }

fn ditherTriangleNoisea(val: f32, pres: i32) -> f32 { return ditherTriangleNoise(vec3f(val),DITHER_TRIANGLENOISE_COORD, pres).r; }
fn ditherTriangleNoise3b(color: vec3f, pres: i32) -> vec3f { return ditherTriangleNoise(color, DITHER_TRIANGLENOISE_COORD, pres); }
fn ditherTriangleNoise4a(color: vec4f, pres: i32) -> vec4f { return vec4f(ditherTriangleNoise(color.rgb, DITHER_TRIANGLENOISE_COORD, pres), color.a); }

fn ditherTriangleNoiseb(val: f32) -> f32 { return ditherTriangleNoise(vec3f(val), DITHER_TRIANGLENOISE_COORD, DITHER_TRIANGLENOISE_PRECISION).r; }
fn ditherTriangleNoise3c(color: vec3f) -> vec3f { return ditherTriangleNoise(color, DITHER_TRIANGLENOISE_COORD, DITHER_TRIANGLENOISE_PRECISION); }
fn ditherTriangleNoise4b(color: vec4f) -> vec4f { return vec4f(ditherTriangleNoise(color.rgb), color.a); }
