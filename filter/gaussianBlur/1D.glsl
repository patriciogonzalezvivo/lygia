#include "../../math/gaussian.glsl"
#include "../../sample/clamp2edge.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: One dimension Gaussian Blur to be applied in two passes
use: gaussianBlur1D(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_direction , const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_TYPE: null
    - GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV): null
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef GAUSSIANBLUR1D_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR1D_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR1D_TYPE vec4
#endif
#endif

#ifndef GAUSSIANBLUR1D_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
#else
#define GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR1D
#define FNC_GAUSSIANBLUR1D

#ifdef PLATFORM_WEBGL

GAUSSIANBLUR1D_TYPE gaussianBlur1D(in SAMPLER_TYPE tex,in vec2 st,in vec2 offset,const int kernelSize){
    GAUSSIANBLUR1D_TYPE accumColor = GAUSSIANBLUR1D_TYPE(0.0);

    float kernelSizef = float(kernelSize);
    float accumWeight = 0.0;
    const float k = 0.39894228;// 1 / sqrt(2*PI)
    for (int i = 0; i < 16; i++) {
        if( i >= kernelSize)
            break;
        float x = -0.5 * (float(kernelSize) - 1.0)+float(i);
        float weight = (k/float(kernelSize)) * gaussian(x, kernelSizef);
        GAUSSIANBLUR1D_TYPE tex = GAUSSIANBLUR1D_SAMPLER_FNC(tex, st + x * offset);
        accumColor += weight * tex;
        accumWeight += weight;
    }
    return accumColor/accumWeight;
}

#else

GAUSSIANBLUR1D_TYPE gaussianBlur1D(in SAMPLER_TYPE tex,in vec2 st,in vec2 offset,const int kernelSize){
    GAUSSIANBLUR1D_TYPE accumColor=GAUSSIANBLUR1D_TYPE(0.);

    float kernelSizef = float(kernelSize);
    
    float accumWeight = 0.0;
    const float k = 0.39894228;// 1 / sqrt(2*PI)
    for (int i = 0; i < kernelSize; i++) {
        float x = -0.5 * ( kernelSizef -1.0) + float(i);
        float weight = (k / kernelSizef) * gaussian(x, kernelSizef);
        GAUSSIANBLUR1D_TYPE tex = GAUSSIANBLUR1D_SAMPLER_FNC(tex, st + x * offset);
        accumColor += weight * tex;
        accumWeight += weight;
    }
    return accumColor/accumWeight;
}
#endif

#endif
