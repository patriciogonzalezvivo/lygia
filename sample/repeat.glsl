#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a repeat wrapping texture
use: <vec4> sampleRepeat(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEREPEAT
#define FNC_SAMPLEREPEAT
vec4 sampleRepeat(SAMPLER_TYPE tex, vec2 st) { 
    return SAMPLER_FNC( tex, fract(st) ); 
}
#endif