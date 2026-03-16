/*
contributors: Patricio Gonzalez Vivo
description: Set of dither methods
use: <vec4|vec3|float> dither(<vec4|vec3|float> value[, <float> time])
options:
    - DITHER_FNC
    - RESOLUTION: view port resolution
    - BLUENOISE_TEXTURE_RESOLUTION
    - BLUENOISE_TEXTURE
    - DITHER_TIME: followed with an elapsed seconds uniform
    - DITHER_CHROMATIC
    - DITHER_PRECISION
    - SAMPLER_FNC
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "dither/interleavedGradientNoise.wgsl"
#include "dither/vlachos.wgsl"
#include "dither/triangleNoise.wgsl"
#include "dither/blueNoise.wgsl"
#include "dither/shift.wgsl"
#include "dither/bayer.wgsl"

// #define DITHER_FNC ditherInterleavedGradientNoise
// #define DITHER_FNC ditherVlachos

fn dither(v: f32) -> f32 { return DITHER_FNC(v); }
fn dither3(v: vec3f) -> vec3f { return DITHER_FNC(v); }
fn dither4(v: vec4f) -> vec4f { return DITHER_FNC(v); }
