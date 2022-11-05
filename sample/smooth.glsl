/*
original_author: Inigo Quiles
description: avoid the ugly artifacts of bilinear texture filtering. You can find more information here https://iquilezles.org/articles/texture
use:> <vec4 sampleSmooth(<sampler2D> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_SAMPLESMOOTH
#define FNC_SAMPLESMOOTH
vec4 sampleSmooth(sampler2D tex, vec2 st, vec2 texResolution) {
    st *= texResolution + 0.5;
    vec2 ist = floor( st );
    vec2 fst = fract( st );
    st = ist + fst*fst*(3.0-2.0*fst); // fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);
    st = (st - 0.5) / texResolution;
    return SAMPLER_FNC( tex, st );
}
#endif