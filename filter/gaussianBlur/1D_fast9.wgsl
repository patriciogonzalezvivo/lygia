#include "../../sample/clamp2edge.wgsl"

/*
contributors: Matt DesLauriers
description: Adapted versions of gaussian fast blur 13 from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur1D_fast9(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_direction)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_FAST9_TYPE
    - GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(TEX, UV)
*/

// #define GAUSSIANBLUR1D_FAST9_TYPE GAUSSIANBLUR_TYPE
// #define GAUSSIANBLUR1D_FAST9_TYPE vec4

// #define GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
// #define GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

GAUSSIANBLUR1D_FAST9_TYPE gaussianBlur1D_fast9(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    GAUSSIANBLUR1D_FAST9_TYPE color = GAUSSIANBLUR1D_FAST9_TYPE(0.);
    let off1 = vec2f(1.3846153846) * offset;
    let off2 = vec2f(3.2307692308) * offset;
    color += GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(tex, st) * .2270270270;
    color += GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(tex, st + (off1)) * .3162162162;
    color += GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(tex, st - (off1)) * .3162162162;
    color += GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(tex, st + (off2)) * .0702702703;
    color += GAUSSIANBLUR1D_FAST9_SAMPLER_FNC(tex, st - (off2)) * .0702702703;
    return color;
}
