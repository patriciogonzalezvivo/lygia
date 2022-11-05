#include "lygia/math/decimation.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: fakes a nearest sample
use:> <vec4 sampleNear(<sampler2D> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_SAMPLENEAR
#define FNC_SAMPLENEAR
vec4 sampleNear(sampler2D tex, vec2 st, vec2 texResolution) {
    vec2 half_pixel = 0.5/texResolution;
    return SAMPLER_FNC( tex, decimation(st, texResolution) + half_pixel );
}
#endif