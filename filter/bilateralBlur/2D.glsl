#include "../../color/space/rgb2luma.glsl"
#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: | 
    two dimensional bilateral Blur, to do it in one single pass.
    Other examples https://www.shadertoy.com/view/4dfGDH , https://www.shadertoy.com/view/XtVGWG
use: bilateralBlur2D(<sampler2D> texture, <vec2> st, <vec2> offset, <int> kernelSize)
options:
    - BILATERALBLUR2D_TYPE: default is vec3
    - BILATERALBLUR2D_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
    - BILATERALBLUR2D_LUMA(RGB): default rgb2luma
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
examples:
    - /shaders/filter_bilateralBlur2D.frag
*/

#ifndef BILATERALBLUR2D_TYPE
#ifdef BILATERALBLUR_TYPE
#define BILATERALBLUR2D_TYPE BILATERALBLUR_TYPE
#else
#define BILATERALBLUR2D_TYPE vec4
#endif
#endif

#ifndef BILATERALBLUR2D_SAMPLER_FNC
#ifdef BILATERALBLUR_SAMPLER_FNC
#define BILATERALBLUR2D_SAMPLER_FNC(TEX, UV) BILATERALBLUR_SAMPLER_FNC(TEX, UV)
#else
#define BILATERALBLUR2D_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef BILATERALBLUR2D_LUMA
#define BILATERALBLUR2D_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#ifndef FNC_BILATERALBLUR2D
#define FNC_BILATERALBLUR2D
BILATERALBLUR2D_TYPE bilateralBlur2D(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
    BILATERALBLUR2D_TYPE accumColor = BILATERALBLUR2D_TYPE(0.);

    #ifndef BILATERALBLUR2D_KERNELSIZE
    #if defined(PLATFORM_WEBGL)
    #define BILATERALBLUR2D_KERNELSIZE 20
    float kernelSizef = float(kernelSize);
    #else
    #define BILATERALBLUR2D_KERNELSIZE kernelSize
    float kernelSizef = float(BILATERALBLUR2D_KERNELSIZE);
    #endif
    #else 
    float kernelSizef = float(BILATERALBLUR2D_KERNELSIZE);
    #endif
    
    float accumWeight = 0.;
    const float k = .15915494; // 1. / (2.*PI)
    const float k2 = k * k;
    
    float kernelSize2 = kernelSizef * kernelSizef;
    BILATERALBLUR2D_TYPE tex0 = BILATERALBLUR2D_SAMPLER_FNC(tex, st);
    float lum0 = BILATERALBLUR2D_LUMA(tex0);

    for (int j = 0; j < BILATERALBLUR2D_KERNELSIZE; j++) {
        #if defined(PLATFORM_WEBGL)
        if (j >= kernelSize)
            break;
        #endif
        float dy = -.5 * (kernelSizef - 1.0) + float(j);
        for (int i = 0; i < BILATERALBLUR2D_KERNELSIZE; i++) {
            #if defined(PLATFORM_WEBGL)
            if (i >= kernelSize)
                break;
            #endif
            float dx = -.5 * (kernelSizef - 1.0) + float(i);
            BILATERALBLUR2D_TYPE t = BILATERALBLUR2D_SAMPLER_FNC(tex, st + vec2(dx, dy) * offset);
            float lum = BILATERALBLUR2D_LUMA(t);
            float dl = 255. * (lum - lum0);
            float weight = (k2 / kernelSize2) * exp(-(dx * dx + dy * dy + dl * dl) / (2. * kernelSize2));
            accumColor += weight * t;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
