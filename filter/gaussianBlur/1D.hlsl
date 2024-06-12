#include "../../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: One dimension Gaussian Blur to be applied in two passes
use: gaussianBlur1D(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel_direction, const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_TYPE: null
    - GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV): null
    - GAUSSIANBLUR1D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef GAUSSIANBLUR1D_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR1D_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR1D_TYPE float4
#endif
#endif

#ifndef GAUSSIANBLUR1D_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
#else
#define GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR1D
#define FNC_GAUSSIANBLUR1D

GAUSSIANBLUR1D_TYPE gaussianBlur1D(in SAMPLER_TYPE tex,in float2 st,in float2 offset,const int kernelSize){
    GAUSSIANBLUR1D_TYPE accumColor = float4(0.0, 0.0, 0.0, 0.0);
    
    float accumWeight = 0.0;
    const float k = 0.39894228;// 1 / sqrt(2*PI)
    float kernelSize2 = float(kernelSize)*float(kernelSize);
    for(int i = 0; i < 16; i++){
        if( i >= kernelSize)
            break;
        float x = -0.5 * (float(kernelSize) - 1.0)+float(i);
        float weight = (k/float(kernelSize)) * exp(-(x*x)/(2.0*kernelSize2));
        accumColor += weight * GAUSSIANBLUR1D_SAMPLER_FNC(tex, st + x * offset);
        accumWeight += weight;
    }
    return accumColor/accumWeight;
}

#endif
