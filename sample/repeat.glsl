#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: fakes a repeat wrapping texture 
use: <vec4> sampleRepeat(<sampler2D> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEREPEAT
#define FNC_SAMPLEREPEAT
vec4 sampleRepeat(sampler2D tex, vec2 st) { 
    return SAMPLER_FNC( tex, fract(st) ); 
}
#endif