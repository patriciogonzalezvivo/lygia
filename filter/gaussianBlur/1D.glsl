#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: one dimension Gaussian Blur to be applied in two passes
use: gaussianBlur1D(<sampler2D> texture, <vec2> st, <vec2> pixel_direction , const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR1D_TYPE:
    - GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV):
    - GAUSSIANBLUR1D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
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
#define GAUSSIANBLUR1D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR1D
#define FNC_GAUSSIANBLUR1D

#ifdef PLATFORM_WEBGL

GAUSSIANBLUR1D_TYPE gaussianBlur1D(in sampler2D tex,in vec2 st,in vec2 offset,const int kernelSize){
    GAUSSIANBLUR1D_TYPE accumColor = GAUSSIANBLUR1D_TYPE(0.0);
    
    float accumWeight = 0.0;
    const float k = 0.39894228;// 1 / sqrt(2*PI)
    float kernelSize2 = float(kernelSize)*float(kernelSize);
    for (int i = 0; i < 16; i++) {
        if( i >= kernelSize)
            break;
        float x = -0.5 * (float(kernelSize) - 1.0)+float(i);
        float weight = (k/float(kernelSize)) * exp(-(x*x)/(2.0*kernelSize2));
        GAUSSIANBLUR1D_TYPE tex = GAUSSIANBLUR1D_SAMPLER_FNC(tex, st + x * offset);
        accumColor += weight * tex;
        accumWeight += weight;
    }
    return accumColor/accumWeight;
}

#else
GAUSSIANBLUR1D_TYPE gaussianBlur1D(in sampler2D tex,in vec2 st,in vec2 offset,const int kernelSize){
    GAUSSIANBLUR1D_TYPE accumColor=GAUSSIANBLUR1D_TYPE(0.);

    #ifndef GAUSSIANBLUR1D_KERNELSIZE
    
    #if defined(PLATFORM_WEBGL)
    #define GAUSSIANBLUR1D_KERNELSIZE 20
    float kernelSizef = float(kernelSize);
    #else
    #define GAUSSIANBLUR1D_KERNELSIZE kernelSize
    float kernelSizef = float(GAUSSIANBLUR1D_KERNELSIZE);
    #endif

    #else
    float kernelSizef = float(GAUSSIANBLUR1D_KERNELSIZE);
    #endif
    
    float accumWeight = 0.0;
    const float k = 0.39894228;// 1 / sqrt(2*PI)
    float kernelSize2= kernelSizef * kernelSizef;
    for (int i = 0; i < GAUSSIANBLUR1D_KERNELSIZE; i++) {
        #if defined(PLATFORM_WEBGL)
        if (i >= kernelSize)
            break;
        #endif
        float x = -0.5 * ( kernelSizef -1.0) + float(i);
        float weight = (k / kernelSizef) * exp(-(x*x)/(2.0*kernelSize2));
        GAUSSIANBLUR1D_TYPE tex = GAUSSIANBLUR1D_SAMPLER_FNC(tex, st + x * offset);
        accumColor += weight * tex;
        accumWeight += weight;
    }
    return accumColor/accumWeight;
}
#endif

#endif
