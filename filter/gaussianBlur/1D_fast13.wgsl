#include "../../sample/clamp2edge.wgsl"

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

// #define GAUSSIANBLUR1D_FAST13_TYPE GAUSSIANBLUR_TYPE
// #define GAUSSIANBLUR1D_FAST13_TYPE vec4

// #define GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
// #define GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

GAUSSIANBLUR1D_FAST13_TYPE gaussianBlur1D_fast13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    GAUSSIANBLUR1D_FAST13_TYPE color = GAUSSIANBLUR1D_FAST13_TYPE(0.);
    let off1 = vec2f(1.411764705882353) * offset;
    let off2 = vec2f(3.2941176470588234) * offset;
    let off3 = vec2f(5.176470588235294) * offset;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st) * .1964825501511404;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off1)) * .2969069646728344;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off1)) * .2969069646728344;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off2)) * .09447039785044732;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off2)) * .09447039785044732;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st + (off3)) * .010381362401148057;
    color += GAUSSIANBLUR1D_FAST13_SAMPLER_FNC(tex, st - (off3)) * .010381362401148057;
    return color;
}
