#include "../sampler.glsl"

/*
contributors: Brad Larson
description: Adapted version of mean average sampling on four coorners of a sampled point from https://github.com/BradLarson/GPUImage2
use: mean(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - MEAN_TYPE: defaults to vec4
    - AVERAGE_SAMPLER_FNC(TEX, UV): defaults to texture2D(tex,TEX, UV)
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef MEAN_TYPE
#define MEAN_TYPE vec4
#endif

#ifndef MEAN_AMOUNT
#define MEAN_AMOUNT mean4
#endif

#ifndef MEAN_SAMPLER_FNC
#define MEAN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_AVERAGE
#define FNC_AVERAGE
MEAN_TYPE mean4(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel) {
    MEAN_TYPE topLeft = MEAN_SAMPLER_FNC(tex, st - pixel);
    MEAN_TYPE bottomLeft = MEAN_SAMPLER_FNC(tex, st + vec2(-pixel.x, pixel.y));
    MEAN_TYPE topRight = MEAN_SAMPLER_FNC(tex, st + vec2(pixel.x, -pixel.y));
    MEAN_TYPE bottomRight = MEAN_SAMPLER_FNC(tex, st + pixel);
    return 0.25 * (topLeft + topRight + bottomLeft + bottomRight);
}

MEAN_TYPE mean(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel) {
    return MEAN_AMOUNT(tex, st, pixel);
}
#endif
