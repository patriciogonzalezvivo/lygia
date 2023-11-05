#include "../../sample/clamp2edge.glsl"

/*
contributors: Brad Larson
description: |
    Adapted version of Sobel edge detection from https://github.com/BradLarson/GPUImage2.
use: edgeSobel(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixels_scale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EDGESOBEL_TYPE: Return type, defaults to float
    - EDGESOBEL_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
examples:
    - /shaders/filter_edge2D.frag
*/

#ifndef EDGESOBEL_TYPE
#ifdef EDGE_TYPE
#define EDGESOBEL_TYPE EDGE_TYPE
#else
#define EDGESOBEL_TYPE float
#endif
#endif

#ifndef EDGESOBEL_SAMPLER_FNC
#ifdef EDGE_SAMPLER_FNC
#define EDGESOBEL_SAMPLER_FNC(TEX, UV) EDGE_SAMPLER_FNC(TEX, UV)
#else
#define EDGESOBEL_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV).r
#endif
#endif

#ifndef FNC_EDGESOBEL
#define FNC_EDGESOBEL
EDGESOBEL_TYPE edgeSobel(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    // get samples around pixel
    EDGESOBEL_TYPE tleft = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(-offset.x, offset.y));
    EDGESOBEL_TYPE left = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(-offset.x, 0.));
    EDGESOBEL_TYPE bleft = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(-offset.x, -offset.y));
    EDGESOBEL_TYPE top = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(0., offset.y));
    EDGESOBEL_TYPE bottom = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(0., -offset.y));
    EDGESOBEL_TYPE tright = EDGESOBEL_SAMPLER_FNC(tex, st + offset);
    EDGESOBEL_TYPE right = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(offset.x, 0.));
    EDGESOBEL_TYPE bright = EDGESOBEL_SAMPLER_FNC(tex, st + vec2(offset.x, -offset.y));
    EDGESOBEL_TYPE x = tleft + 2. * left + bleft - tright - 2. * right - bright;
    EDGESOBEL_TYPE y = -tleft - 2. * top - tright + bleft + 2. * bottom + bright;
    return sqrt((x * x) + (y * y));
}
#endif
