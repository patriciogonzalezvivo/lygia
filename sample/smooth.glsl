#include "../math/cubic.glsl"
#include "../math/invCubic.glsl"
#include "../math/quintic.glsl"

/*
original_author: Inigo Quiles
description: avoid the ugly artifacts of bilinear texture filtering. You can find more information here https://iquilezles.org/articles/texture
use:> <vec4 sampleSmooth(<sampler2D> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLESMOOTH_POLYNOMIAL: cubic or quartic
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef SAMPLESMOOTH_POLYNOMIAL
#define SAMPLESMOOTH_POLYNOMIAL cubic or quintic
#endif

#ifndef FNC_SAMPLESMOOTH
#define FNC_SAMPLESMOOTH
vec4 sampleSmooth(sampler2D tex, vec2 st, vec2 texResolution) {
    st *= texResolution + 0.5;
    vec2 fst = fract( st );
    st = floor( st );
    st += SAMPLESMOOTH_POLYNOMIAL(fst);
    st = (st - 0.5) / texResolution;
    return SAMPLER_FNC( tex, st );
}
#endif