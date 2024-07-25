#include "../math/mod.wgsl"

/*
contributors:
    - Patricio Gonzalez Vivo
    - Stevan Dedovic
description: Signed Random
use: srandomX(<vec2|vec3> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SRANDOM
#define FNC_SRANDOM

fn srandom(x: f32) -> f32 {
  return -1. + 2. * fract(sin(x) * 43758.5453);
}

fn srandom2(st: vec2f) -> f32 {
  return -1. + 2. * fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

fn srandom3(pos: vec3f) -> f32 {
  return -1. + 2. * fract(sin(dot(pos.xyz, vec3(70.9898, 78.233, 32.4355))) * 43758.5453123);
}

fn srandom4(pos: vec4f) -> f32{
    let dot_product = dot(pos, vec4(12.9898,78.233,45.164,94.673));
    return -1. + 2. * fract(sin(dot_product) * 43758.5453);
}

fn srandom22(st: vec2f) -> vec2f {
    let k = vec2(.3183099, .3678794);
    let tmp = st * k + k.yx;
    return -1. + 2. * fract(16. * k * fract(tmp.x * tmp.y * (tmp.x + tmp.y)));
}

fn srandom33(p: vec3f) -> vec3f {
    let tmp = vec3( dot(p, vec3(127.1, 311.7, 74.7)),
            dot(p, vec3(269.5, 183.3, 246.1)),
            dot(p, vec3(113.5, 271.9, 124.6)));
    return -1. + 2. * fract(sin(tmp) * 43758.5453123);
}

fn srandom_tile2(p: vec2f, tileLength: f32) -> vec2f {
    let tmp = mod2(p, vec2(tileLength));
    return srandom22(tmp);
}

fn srandom_tile3(p: vec3f, tileLength: f32) -> vec3f {
    let tmp = mod3(p, vec3(tileLength));
    return srandom33(tmp);
}

#endif