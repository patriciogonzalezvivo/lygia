#include "../math/gaussian.hlsl"
#include "../sample/clamp2edge.hlsl"
#include "../color/space/rgb2luma.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    This is a two dimensioanl Bilateral filter (for a single pass) It's a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a weighted average of
    intensity values from nearby pixels. This filter is very effective at noise removal while
    preserving edges. It is very similar to the Gaussian blur, but it also takes into account the
    intensity differences between a pixel and its neighbors. This is what makes it particularly
    effective at noise removal while preserving edges.
use: bilateral(<SAMPLER_TYPE> texture, <float2> st, <float2> duv)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERAL_AMOUNT
    - BILATERAL_TYPE
    - BILATERAL_SAMPLER_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BILATERAL_TYPE
#define BILATERAL_TYPE float4
#endif

#ifndef BILATERAL_SAMPLER_FNC
#define BILATERAL_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
#endif

#ifndef BILATERAL_LUMA
#define BILATERAL_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#ifndef BILATERAL_KERNEL_MAXSIZE
#define BILATERAL_KERNEL_MAXSIZE 20
#endif

#ifndef FNC_BILATERALFILTER
#define FNC_BILATERALFILTER

BILATERAL_TYPE bilateral(SAMPLER_TYPE tex, float2 st, float2 offset, const int kernelSize) {
    BILATERAL_TYPE accumColor = float4(0.0, 0.0, 0.0, 0.0);
    float accumWeight = 0.0;
    const float k = 0.15915494; // 1. / (2.*PI)
    const float k2 = k * k;
    float kernelSizef = float(kernelSize);
    float kernelSize2f = kernelSizef * kernelSizef;
    BILATERAL_TYPE tex0 = BILATERAL_SAMPLER_FNC(tex, st);
    float lum0 = BILATERAL_LUMA(tex0);

    for (int j = 0; j < BILATERAL_KERNEL_MAXSIZE; j++) {
        if (j >= kernelSize)
            break;
        float dy = -0.5 * (kernelSizef - 1.0) + float(j);
        for (int i = 0; i < BILATERAL_KERNEL_MAXSIZE; i++) {
            if (i >= kernelSize)
                break;
            float dx = -0.5 * (kernelSizef - 1.0) + float(i);

            BILATERAL_TYPE t = BILATERAL_SAMPLER_FNC(tex, st + float2(dx, dy) * offset);
            float lum = BILATERAL_LUMA(t);
            float dl = 255.0 * (lum - lum0);
            float weight = (k2 / kernelSize2f) * gaussian(float3(dx, dy, dl), kernelSizef);
            accumColor += weight * t;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
