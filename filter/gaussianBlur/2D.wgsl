#include "../../math/gaussian.wgsl"
#include "../../sample/clamp2edge.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Two dimension Gaussian Blur to be applied in only one passes
use: gaussianBlur2D(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_direction, const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR2D_TYPE: Default `vec4`
    - GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV): Default `texture2D(tex, TEX, UV)`
    - GAUSSIANBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example  RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
examples:
    - /shaders/filter_gaussianBlur2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define GAUSSIANBLUR2D_TYPE GAUSSIANBLUR_TYPE
// #define GAUSSIANBLUR2D_TYPE vec4

// #define GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
// #define GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

GAUSSIANBLUR2D_TYPE gaussianBlur2D(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset, const int kernelSize) {
    GAUSSIANBLUR2D_TYPE accumColor = GAUSSIANBLUR2D_TYPE(0.);
    
    
const GAUSSIANBLUR2D_KERNELSIZE: f32 = 20;
            let kernelSizef = float(kernelSize);
//             #define GAUSSIANBLUR2D_KERNELSIZE kernelSize
            let kernelSizef = float(GAUSSIANBLUR2D_KERNELSIZE);

        let kernelSizef = float(GAUSSIANBLUR2D_KERNELSIZE);

    let accumWeight = 0.;
    const float k = 0.15915494; // 1 / (2*PI)
    let xy = vec2f(0.0);
    for (int j = 0; j < GAUSSIANBLUR2D_KERNELSIZE; j++) {
        if (j >= kernelSize)
            break;
        xy.y = -.5 * (kernelSizef - 1.) + float(j);
        for (int i = 0; i < GAUSSIANBLUR2D_KERNELSIZE; i++) {
            if (i >= kernelSize)
                break;
            xy.x = -0.5 * (kernelSizef - 1.) + float(i);
            let weight = (k / kernelSizef) * gaussian(xy, kernelSizef);
            accumColor += weight * GAUSSIANBLUR2D_SAMPLER_FNC(tex, st + xy * offset);
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
