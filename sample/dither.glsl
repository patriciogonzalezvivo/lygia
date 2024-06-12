#include "nearest.glsl"
#include "../color/dither.glsl"

#include "../color/space/gamma2linear.glsl"
#include "../color/space/linear2gamma.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a texture and dither using a 8x8 Bayer matrix
use: <vec3> ditherBayer(<SAMPLER_TYPE> tex, <vec2> st, <vec2> resolution)
options:
    - DITHERBAKER_LUT(COLOR): function that returns a vec3 with the color to use for the dithering
examples:
    - /shaders/color_dither_bayer.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLEDITHER_FNC
#define SAMPLEDITHER_FNC ditherBayer
#endif

#ifndef FNC_SAMPLEDITHER
#define FNC_SAMPLEDITHER
vec4 sampleDither(SAMPLER_TYPE tex, const in vec2 st, const in vec2 resolution) {
    vec4 color = sampleNearest(tex, st, resolution);
    gamma2linear(color);
    color = SAMPLEDITHER_FNC(color, st * resolution);
    linear2gamma(color);
    return color;
}
#endif