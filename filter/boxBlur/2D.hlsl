/*
original_author: Patricio Gonzalez Vivo
description: simple two dimentional box blur, so can be apply in a single pass
use: boxBlur2D(<sampler2D> texture, <float2> st, <float2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR2D_TYPE: Default `float4`
    - BOXBLUR2D_SAMPLER_FNC(POS_UV): default is `texture2D(tex, POS_UV)`
    - BOXBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef BOXBLUR2D_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR2D_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR2D_TYPE float4
#endif
#endif

#ifndef BOXBLUR2D_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR2D_SAMPLER_FNC(POS_UV) BOXBLUR_SAMPLER_FNC(POS_UV)
#else
#define BOXBLUR2D_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef BOXBLUR2D_KERNEL_MAXSIZE
#define BOXBLUR2D_KERNEL_MAXSIZE 20
#endif

#ifndef FNC_BOXBLUR2D
#define FNC_BOXBLUR2D
BOXBLUR2D_TYPE boxBlur2D(in sampler2D tex, in float2 st, in float2 pixel, const int kernelSize) {
    BOXBLUR2D_TYPE color = float4(0.0, 0.0, 0.0, 0.0);
    

    float accumWeight = 0.;
    float f_kernelSize = float(kernelSize);
    float kernelSize2 = f_kernelSize * f_kernelSize;
    float weight = 1. / kernelSize2;

    for (int j = 0; j < BOXBLUR2D_KERNEL_MAXSIZE; j++) {
        if (j >= kernelSize)
            break;
            
        float y = -.5 * (f_kernelSize - 1.) + float(j);
        for (int i = 0; i < BOXBLUR2D_KERNEL_MAXSIZE; i++) {
            if (i >= kernelSize)
                break;

            float x = -.5 * (f_kernelSize - 1.) + float(i);
            color += BOXBLUR2D_SAMPLER_FNC(st + float2(x, y) * pixel) * weight;
        }
    }
    return color;
}
#endif
