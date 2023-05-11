#include "nearest.glsl"
#include "../color/dither/bayer.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: sample a texture and dither using a 8x8 Bayer matrix
use: <vec3> ditherBayer(<sampler2D> tex, <vec2> st, <vec2> resolution)
options:
    - DITHERBAKER_LUT(COLOR): function that returns a vec3 with the color to use for the dithering
examples:
    - /shaders/color_dither_bayer.frag
*/

#ifndef FNC_SAMPLEDITHER
#define FNC_SAMPLEDITHER
vec3 sampleDither(sampler2D tex, const in vec2 st, const in vec2 resolution) {
    return ditherBayer(sampleNearest(tex, st, resolution).rgb, st * resolution);
}
#endif