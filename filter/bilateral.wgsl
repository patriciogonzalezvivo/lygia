#include "../math/gaussian.wgsl"
#include "../color/space/rgb2luma.wgsl"
#include "../sample/clamp2edge.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    This is a two dimensioanl Bilateral filter (for a single pass) It's a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a weighted average of
    intensity values from nearby pixels. This filter is very effective at noise removal while
    preserving edges. It is very similar to the Gaussian blur, but it also takes into account the
    intensity differences between a pixel and its neighbors. This is what makes it particularly
    effective at noise removal while preserving edges.
use: bilateral(<SAMPLER_TYPE> texture, <vec2> st, <vec2> duv [, <int> kernelSize]])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERAL_TYPE: default is vec3
    - BILATERAL_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
    - BILATERAL_LUMA(RGB): default rgb2luma
    - BILATERAL_KERNEL_MAXSIZE: default 20
examples:
    - /shaders/filter_bilateral2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define BILATERAL_TYPE vec4

// #define BILATERAL_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

// #define BILATERAL_LUMA(RGB) rgb2luma(RGB.rgb)

BILATERAL_TYPE bilateral(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset, const int kernelSize) {
    BILATERAL_TYPE accumColor = BILATERAL_TYPE(0.);

const BILATERAL_KERNELSIZE: f32 = 20;
    let kernelSizef = float(kernelSize);
//     #define BILATERAL_KERNELSIZE kernelSize
    let kernelSizef = float(BILATERAL_KERNELSIZE);
    let kernelSizef = float(BILATERAL_KERNELSIZE);
    
    let accumWeight = 0.0;
    const float k = 0.15915494; // 1. / (2.*PI)
    let k2 = k * k;
    
    let kernelSize2 = kernelSizef * kernelSizef;
    BILATERAL_TYPE tex0 = BILATERAL_SAMPLER_FNC(tex, st);
    let lum0 = BILATERAL_LUMA(tex0);

    for (int j = 0; j < BILATERAL_KERNELSIZE; j++) {
        if (j >= kernelSize)
            break;
        let dy = -0.5 * (kernelSizef - 1.0) + float(j);
        for (int i = 0; i < BILATERAL_KERNELSIZE; i++) {
            if (i >= kernelSize)
                break;
            let dx = -0.5 * (kernelSizef - 1.0) + float(i);
            BILATERAL_TYPE t = BILATERAL_SAMPLER_FNC(tex, st + vec2f(dx, dy) * offset);
            let lum = BILATERAL_LUMA(t);
            let dl = 255.0 * (lum - lum0);
            let weight = (k2 / kernelSize2) * gaussian(vec3f(dx,dy,dl), kernelSizef);
            accumColor += weight * t;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
