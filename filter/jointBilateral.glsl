#include "../sample/clamp2edge.glsl"
#include "../math/gaussian.glsl"

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
    - JOINTBILATERAL_TYPEGUIDE: defualts to vec3
    - JOINTBILATERAL_SAMPLEGUIDE_FNC(TEX, UV): defaults to sampleClamp2edge(TEX, UV).rgb
    - JOINTBILATERAL_KERNELSIZE: defaults to  9
    - JOINTBILATERAL_INTENSITY_SIGMA: defaults to 0.026
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef JOINTBILATERAL_TYPE
#define JOINTBILATERAL_TYPE vec4
#endif

#ifndef JOINTBILATERAL_SAMPLE_FNC
#define JOINTBILATERAL_SAMPLE_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
#endif

#ifndef JOINTBILATERAL_TYPEGUIDE
#define JOINTBILATERAL_TYPEGUIDE vec3
#endif

#ifndef JOINTBILATERAL_SAMPLEGUIDE_FNC
#define JOINTBILATERAL_SAMPLEGUIDE_FNC(TEX, UV) sampleClamp2edge(TEX, UV).rgb
#endif

// #ifndef JOINTBILATERAL_KERNELSIZE
// #define JOINTBILATERAL_KERNELSIZE 9
// #endif

#ifndef JOINTBILATERAL_INTENSITY_SIGMA
#define JOINTBILATERAL_INTENSITY_SIGMA 0.026
#endif

#ifndef FNC_JOINTBILATERAL
#define FNC_JOINTBILATERAL

JOINTBILATERAL_TYPE jointBilateral(SAMPLER_TYPE tex, SAMPLER_TYPE guide, vec2 uv, vec2 pixel, const int kernelSize) {
    JOINTBILATERAL_TYPEGUIDE centerGuide = JOINTBILATERAL_SAMPLEGUIDE_FNC(guide, uv);

    #ifndef JOINTBILATERAL_KERNELSIZE
    
    #if defined(PLATFORM_WEBGL)
    #define JOINTBILATERAL_KERNELSIZE 20
    float kernelSizef = float(kernelSize);
    #else
    #define JOINTBILATERAL_KERNELSIZE kernelSize
    float kernelSizef = float(JOINTBILATERAL_KERNELSIZE);
    #endif

    #else
    float kernelSizef = float(JOINTBILATERAL_KERNELSIZE);
    #endif

    JOINTBILATERAL_TYPE sum = JOINTBILATERAL_TYPE(0.0);
    float weight = 0.0;
    const float k = 0.15915494; // 1 / (2*PI)
    for (int j = 0; j < JOINTBILATERAL_KERNELSIZE; j++) {
        #if defined(PLATFORM_WEBGL)
        if (j >= kernelSize)
            break;
        #endif
        float y = -.5 * (kernelSizef - 1.) + float(j);
        for (int i = 0; i < JOINTBILATERAL_KERNELSIZE; i++) {
            #if defined(PLATFORM_WEBGL)
            if (i >= kernelSize)
                break;
            #endif
            float x = -.5 * (kernelSizef - 1.) + float(i);

            vec2 xy = uv + vec2(x, y) * pixel;
            JOINTBILATERAL_TYPE sample = JOINTBILATERAL_SAMPLE_FNC(tex, xy);
            JOINTBILATERAL_TYPEGUIDE sampleGuide = JOINTBILATERAL_SAMPLEGUIDE_FNC(guide, xy);

            float w = 1.0;
            // w *= (k / kernelSizef) * exp(-(x * x + y * y) / (2. * kernelSize2));
            w = (k / kernelSizef) * gaussian(vec2(x, y), kernelSizef);
            w *= gaussian(centerGuide - sampleGuide, JOINTBILATERAL_INTENSITY_SIGMA);
            sum += sample * w;
            weight += w;
        }
    }

    return sum / weight;
}

#endif
