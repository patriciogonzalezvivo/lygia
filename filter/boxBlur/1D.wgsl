#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple one dimensional box blur, to be applied in two passes
use: boxBlur1D(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR1D_TYPE: default is vec4
    - BOXBLUR1D_SAMPLER_FNC(TEX, UV): default texture2D(tex, TEX, UV)
    - BOXBLUR1D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define BOXBLUR1D_TYPE BOXBLUR_TYPE
// #define BOXBLUR1D_TYPE vec4

// #define BOXBLUR1D_SAMPLER_FNC(TEX, UV) BOXBLUR_SAMPLER_FNC(TEX, UV)
// #define BOXBLUR1D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

BOXBLUR1D_TYPE boxBlur1D(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset, const int kernelSize) {
    BOXBLUR1D_TYPE color = BOXBLUR1D_TYPE(0.);
    
    
const BOXBLUR1D_KERNELSIZE: f32 = 20;
    let kernelSizef = float(kernelSize);
//     #define BOXBLUR1D_KERNELSIZE kernelSize
    let kernelSizef = float(BOXBLUR1D_KERNELSIZE);

    let kernelSizef = float(BOXBLUR1D_KERNELSIZE);

    let weight = 1. / kernelSizef;
    for (int i = 0; i < BOXBLUR1D_KERNELSIZE; i++) {
        if (i >= kernelSize)
            break;
        let x = -.5 * (kernelSizef - 1.) + float(i);
        color += BOXBLUR1D_SAMPLER_FNC(tex, st + offset * x ) * weight;
    }
    return color;
}
