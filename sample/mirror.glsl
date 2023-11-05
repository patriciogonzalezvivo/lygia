#include "../sample.glsl"
#include "../math/mirror.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a mirror wrapping texture 
use: <vec4> sampleMirror(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
examples:
    - /shaders/sample_wrap_mirror.frag
*/

#ifndef FNC_SAMPLEMIRROR
#define FNC_SAMPLEMIRROR
vec4 sampleMirror(SAMPLER_TYPE tex, vec2 st) { 
    return SAMPLER_FNC( tex, mirror(st) ); 
}
#endif