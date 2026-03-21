#include "../sample/clamp2edge.wgsl"
#include "../math/gaussian.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Joint Bilateral Filter
    This is a joint bilateral filter that uses a 2D gaussian kernel to approximate the
    bilateral filter. It is based on the paper "Fast Bilateral Filtering for the GPU"
    Interesting article about it and their uses: https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/
use: bilateralBlur(<SAMPLER_TYPE> texture, <vec2> st, <vec2> duv)
options:
    - JOINTBILATERAL_TYPE: defaults to vec4
    - JOINTBILATERAL_SAMPLE_FNC(TEX, UV): defaults to sampleClamp2edge(tex, UV)
    - JOINTBILATERAL_TYPEGUIDE: defaults to vec3
    - JOINTBILATERAL_SAMPLEGUIDE_FNC(TEX, UV): defaults to sampleClamp2edge(TEX, UV).rgb
    - JOINTBILATERAL_KERNELSIZE: defaults to  9
    - JOINTBILATERAL_INTENSITY_SIGMA: defaults to 0.026
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define JOINTBILATERAL_TYPE vec4

// #define JOINTBILATERAL_SAMPLE_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

// #define JOINTBILATERAL_TYPEGUIDE vec3

// #define JOINTBILATERAL_SAMPLEGUIDE_FNC(TEX, UV) sampleClamp2edge(TEX, UV).rgb

// #ifndef JOINTBILATERAL_KERNELSIZE
// #define JOINTBILATERAL_KERNELSIZE 9
// #endif

const JOINTBILATERAL_INTENSITY_SIGMA: f32 = 0.026;

JOINTBILATERAL_TYPE jointBilateral(SAMPLER_TYPE tex, SAMPLER_TYPE guide, vec2 uv, vec2 pixel, const int kernelSize) {
    JOINTBILATERAL_TYPEGUIDE centerGuide = JOINTBILATERAL_SAMPLEGUIDE_FNC(guide, uv);

    
const JOINTBILATERAL_KERNELSIZE: f32 = 20;
    let kernelSizef = float(kernelSize);
//     #define JOINTBILATERAL_KERNELSIZE kernelSize
    let kernelSizef = float(JOINTBILATERAL_KERNELSIZE);

    let kernelSizef = float(JOINTBILATERAL_KERNELSIZE);

    JOINTBILATERAL_TYPE sum = JOINTBILATERAL_TYPE(0.0);
    let weight = 0.0;
    const float k = 0.15915494; // 1 / (2*PI)
    for (int j = 0; j < JOINTBILATERAL_KERNELSIZE; j++) {
        if (j >= kernelSize)
            break;
        let y = -.5 * (kernelSizef - 1.) + float(j);
        for (int i = 0; i < JOINTBILATERAL_KERNELSIZE; i++) {
            if (i >= kernelSize)
                break;
            let x = -.5 * (kernelSizef - 1.) + float(i);

            let xy = uv + vec2f(x, y) * pixel;
            JOINTBILATERAL_TYPE sample = JOINTBILATERAL_SAMPLE_FNC(tex, xy);
            JOINTBILATERAL_TYPEGUIDE sampleGuide = JOINTBILATERAL_SAMPLEGUIDE_FNC(guide, xy);

            let w = 1.0;
            // w *= (k / kernelSizef) * exp(-(x * x + y * y) / (2. * kernelSize2));
            w = (k / kernelSizef) * gaussian(vec2f(x, y), kernelSizef);
            w *= gaussian(centerGuide - sampleGuide, JOINTBILATERAL_INTENSITY_SIGMA);
            sum += sample * w;
            weight += w;
        }
    }

    return sum / weight;
}
