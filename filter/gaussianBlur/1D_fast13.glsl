#include "../../sample/clamp2edge.glsl"

/*
function: gaussianBlur1D_fast13
contributors: Matt DesLauriers
description: Adapted versions of gaussian fast blur 13 from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur1D_fast13(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_direction)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_FAST13_TYPE
    - GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(TEX, UV)
*/

#ifndef GAUSSIANBLUR1D_FAST13_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR1D_FAST13_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR1D_FAST13_TYPE vec4
#endif
#endif

#ifndef GAUSSIANBLUR1D_FAST13_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
#else
#define GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR1D_FAST13
#define FNC_GAUSSIANBLUR1D_FAST13
GAUSSIANBLUR1D_FAST13_TYPE gaussianBlur1D_fast13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    GAUSSIANBLUR1D_FAST13_TYPE color = GAUSSIANBLUR1D_FAST13_TYPE(0.);
    vec2 off1 = vec2(1.411764705882353) * offset;
    vec2 off2 = vec2(3.2941176470588234) * offset;
    vec2 off3 = vec2(5.176470588235294) * offset;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st) * .1964825501511404;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off1)) * .2969069646728344;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off1)) * .2969069646728344;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off2)) * .09447039785044732;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off2)) * .09447039785044732;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off3)) * .010381362401148057;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off3)) * .010381362401148057;
    return color;
}
#endif
