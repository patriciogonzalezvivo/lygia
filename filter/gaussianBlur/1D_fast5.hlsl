#include "../../sampler.hlsl"

/*
contributors: Matt DesLauriers
description: Adapted versions of gaussian fast blur 13 from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur1D_fast5(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel_direction)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_FAST5_TYPE
    - GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(POS_UV)
*/

#ifndef GAUSSIANBLUR1D_FAST5_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR1D_FAST5_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR1D_FAST5_TYPE float4
#endif
#endif

#ifndef GAUSSIANBLUR1D_FAST5_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
#else
#define GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR1D_FAST5
#define FNC_GAUSSIANBLUR1D_FAST5
GAUSSIANBLUR1D_FAST5_TYPE gaussianBlur1D_fast5(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
    float2 off1 = float2(1.3333333333333333, 1.3333333333333333) * offset;
    GAUSSIANBLUR1D_FAST5_TYPE color = GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(tex, st) * .29411764705882354;
    color += GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(tex, st + (off1)) * .35294117647058826;
    color += GAUSSIANBLUR1D_FAST5_SAMPLER_FNC(tex, st - (off1)) * .35294117647058826;
    return color;
}
#endif
