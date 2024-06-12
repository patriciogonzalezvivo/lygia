#include "../space/nearest.glsl"
#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a nearest sample
use: <vec4> sampleNearest(<SAMPLER_TYPE> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
examples:
    - /shaders/sample_filter_nearest.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLENEAREST
#define FNC_SAMPLENEAREST
vec4 sampleNearest(SAMPLER_TYPE tex, vec2 st, vec2 texResolution) {
    return SAMPLER_FNC( tex, nearest(st, texResolution) );
}
#endif