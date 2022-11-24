#include "../../color/space/rgb2luma.hlsl"
#include "../../sample.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: two dimensional bilateral Blur, to do it in one single pass
use: bilateralBlur2D(<sampler2D> texture, <float2> st, <float2> offset, <int> kernelSize)
options:
    - BILATERALBLUR2D_TYPE: default is float3
    - BILATERALBLUR2D_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
    - BILATERALBLUR2D_LUMA(RGB): default rgb2luma
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef BILATERALBLUR2D_TYPE
#ifdef BILATERALBLUR_TYPE
#define BILATERALBLUR2D_TYPE BILATERALBLUR_TYPE
#else
#define BILATERALBLUR2D_TYPE float4
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

#ifndef BILATERALBLUR2D_KERNEL_MAXSIZE
#define BILATERALBLUR2D_KERNEL_MAXSIZE 20
#endif

#ifndef FNC_BILATERALBLUR2D
#define FNC_BILATERALBLUR2D
BILATERALBLUR2D_TYPE bilateralBlur2D(sampler2D tex, float2 st, float2 offset, const int kernelSize) {
    BILATERALBLUR2D_TYPE accumColor = float4(0.0, 0.0, 0.0, 0.0);
    float accumWeight = 0.;
    const float k = .15915494; // 1. / (2.*PI)
    const float k2 = k * k;
    float kernelSizef = float(kernelSize);
    float kernelSize2f = kernelSizef * kernelSizef;
    BILATERALBLUR2D_TYPE tex0 = BILATERALBLUR2D_SAMPLER_FNC(tex, st);
    float lum0 = BILATERALBLUR2D_LUMA(tex0);

    for (int j = 0; j < BILATERALBLUR2D_KERNEL_MAXSIZE; j++) {
        if (j >= kernelSize)
            break;
        float dy = -.5 * (kernelSizef - 1.0) + float(j);
        for (int i = 0; i < BILATERALBLUR2D_KERNEL_MAXSIZE; i++) {
            if (i >= kernelSize)
                break;
            float dx = -.5 * (kernelSizef - 1.0) + float(i);

            BILATERALBLUR2D_TYPE t = BILATERALBLUR2D_SAMPLER_FNC(tex, st + float2(dx, dy) * offset );
            float lum = BILATERALBLUR2D_LUMA(t);
            float dl = 255. * (lum - lum0);
            float weight = (k2 / kernelSize2f) * exp(-(dx * dx + dy * dy + dl * dl) / (2. * kernelSize2f));
            accumColor += weight * t;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
