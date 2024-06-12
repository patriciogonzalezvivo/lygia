#include "../math/cubic.glsl"
#include "../math/quintic.glsl"
#include "../sampler.glsl"

/*
contributors: Inigo Quiles
description: avoid the ugly artifacts of bilinear texture filtering. You can find more information here https://iquilezles.org/articles/texture
use: <vec4> sampleSmooth(<SAMPLER_TYPE> tex, <vec2> st, <vec2> texResolution)
options:
    - SAMPLER_FNC(TEX, UV): nan
    - SAMPLESMOOTH_POLYNOMIAL: cubic or quartic
examples:
    - /shaders/sample_filter_nearest.frag
*/

#ifndef SAMPLESMOOTH_POLYNOMIAL
#define SAMPLESMOOTH_POLYNOMIAL cubic
#endif

#ifndef FNC_SAMPLESMOOTH
#define FNC_SAMPLESMOOTH
vec4 sampleSmooth(SAMPLER_TYPE tex, vec2 st, vec2 texResolution) {
    st *= texResolution + 0.5;
    vec2 fst = fract( st );
    st = floor( st );
    st += SAMPLESMOOTH_POLYNOMIAL(fst);
    st = (st - 0.5) / texResolution;
    return SAMPLER_FNC( tex, st );
}
#endif