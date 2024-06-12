#include "../../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple one dimentional box blur, to be applied in two passes
use: boxBlur1D(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR1D_TYPE: default is float4
    - BOXBLUR1D_SAMPLER_FNC(TEX, UV): default texture2D(tex, TEX, UV)
    - BOXBLUR1D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BOXBLUR1D_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR1D_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR1D_TYPE float4
#endif
#endif

#ifndef BOXBLUR1D_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR1D_SAMPLER_FNC(TEX, UV) BOXBLUR_SAMPLER_FNC(TEX, UV)
#else
#define BOXBLUR1D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef FNC_BOXBLUR1D
#define FNC_BOXBLUR1D
BOXBLUR1D_TYPE boxBlur1D(in SAMPLER_TYPE tex, in float2 st, in float2 offset, const int kernelSize) {
    BOXBLUR1D_TYPE color = float4(0.0, 0.0, 0.0, 0.0);
    #ifndef BOXBLUR1D_KERNELSIZE
    #define BOXBLUR1D_KERNELSIZE kernelSize
    #endif

    float f_kernelSize = float(BOXBLUR1D_KERNELSIZE);
    float weight = 1. / f_kernelSize;

    for (int i = 0; i < BOXBLUR1D_KERNELSIZE; i++) {
        float x = -.5 * (f_kernelSize - 1.) + float(i);
        color += BOXBLUR1D_SAMPLER_FNC(tex, st + offset * x ) * weight;
    }
    return color;
}
#endif
