#include "../../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Two dimension Gaussian Blur to be applied in only one passes
use: gaussianBlur2D(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel_direction, const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR2D_TYPE: Default `float4`
    - GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV): Default `texture2D(tex, TEX, UV)`
    - GAUSSIANBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef GAUSSIANBLUR2D_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR2D_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR2D_TYPE float4
#endif
#endif

#ifndef GAUSSIANBLUR2D_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV) GAUSSIANBLUR_SAMPLER_FNC(TEX, UV)
#else
#define GAUSSIANBLUR2D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef GAUSSIANBLUR2D_KERNEL_MAXSIZE
#define GAUSSIANBLUR2D_KERNEL_MAXSIZE 12
#endif

#ifndef FNC_GAUSSIANBLUR2D
#define FNC_GAUSSIANBLUR2D
GAUSSIANBLUR2D_TYPE gaussianBlur2D(in SAMPLER_TYPE tex, in float2 st, in float2 offset, const int kernelSize) {
    GAUSSIANBLUR2D_TYPE accumColor = float4(0.0, 0.0, 0.0, 0.0);

    float accumWeight = 0.;
    const float k = .15915494; // 1 / (2*PI)
    float kernelSizef = float(kernelSize);
    float kernelSize2 = kernelSizef * kernelSizef;

    for (int j = 0; j < GAUSSIANBLUR2D_KERNEL_MAXSIZE; j++) {
        if (j >= kernelSize)
            break;
        float y = -.5 * (kernelSize2 - 1.) + float(j);
        for (int i = 0; i < GAUSSIANBLUR2D_KERNEL_MAXSIZE; i++) {
            if (i >= kernelSize)
                break;
                
            float x = -.5 * (kernelSize2 - 1.) + float(i);
            float weight = (k / kernelSize2) * exp(-(x * x + y * y) / (2. * kernelSize2));
            
            accumColor += weight * GAUSSIANBLUR2D_SAMPLER_FNC(tex, st + float2(x, y) * offset);
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
