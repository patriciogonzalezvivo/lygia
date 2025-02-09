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

#include "dither/interleavedGradientNoise.glsl"
#include "dither/vlachos.glsl"
#include "dither/triangleNoise.glsl"
#include "dither/blueNoise.glsl"
#include "dither/shift.glsl"
#include "dither/bayer.glsl"

#ifndef DITHER_FNC
#ifdef TARGET_MOBILE
#define DITHER_FNC ditherInterleavedGradientNoise
#else
#define DITHER_FNC ditherVlachos
#endif
#endif

#ifndef FNC_DITHER
#define FNC_DITHER

float dither(float v) { return DITHER_FNC(v); }
vec3 dither(vec3 v) { return DITHER_FNC(v); }
vec4 dither(vec4 v) { return DITHER_FNC(v); }

#endif