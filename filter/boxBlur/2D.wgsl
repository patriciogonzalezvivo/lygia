#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple two dimensional box blur, so can be apply in a single pass
use: boxBlur2D(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR2D_TYPE: Default `vec4`
    - BOXBLUR2D_SAMPLER_FNC(TEX, UV): default is `texture2D(tex, TEX, UV)`
    - BOXBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
examples:
    - /shaders/filter_boxBlur2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define BOXBLUR2D_TYPE BOXBLUR_TYPE
// #define BOXBLUR2D_TYPE vec4

// #define BOXBLUR2D_SAMPLER_FNC(TEX, UV) BOXBLUR_SAMPLER_FNC(TEX, UV)
// #define BOXBLUR2D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

BOXBLUR2D_TYPE boxBlur2D(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, const int kernelSize) {
    BOXBLUR2D_TYPE color = BOXBLUR2D_TYPE(0.);

const BOXBLUR2D_KERNELSIZE: f32 = 20;
    let kernelSizef = float(kernelSize);
//     #define BOXBLUR2D_KERNELSIZE kernelSize
    let kernelSizef = float(BOXBLUR2D_KERNELSIZE);

    let kernelSizef = float(BOXBLUR2D_KERNELSIZE);

    let accumWeight = 0.;
    let kernelSize2 = kernelSizef * kernelSizef;
    let weight = 1. / kernelSize2;

    for (int j = 0; j < BOXBLUR2D_KERNELSIZE; j++) {
        if (j >= kernelSize)
            break;
        let y = -.5 * (kernelSizef - 1.) + float(j);
        for (int i = 0; i < BOXBLUR2D_KERNELSIZE; i++) {
            if (i >= kernelSize)
                break;
            let x = -.5 * (kernelSizef - 1.) + float(i);
            color += BOXBLUR2D_SAMPLER_FNC(tex, st + vec2f(x, y) * pixel) * weight;
        }
    }
    return color;
}
