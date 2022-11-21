#include "../space/nearest.glsl"
#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: fakes a nearest sample
use: <vec4> sampleNearest(<sampler2D> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLENEAREST
#define FNC_SAMPLENEAREST
vec4 sampleNearest(sampler2D tex, vec2 st, vec2 texResolution) {
    return SAMPLER_FNC( tex, nearest(st, texResolution) );
}
#endif