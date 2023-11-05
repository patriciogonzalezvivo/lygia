#include "../math/gaussian.hlsl"
#include "../sample/clamp2edge.hlsl"
#include "../color/space/rgb2luma.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    The bilateral filter is a non-linear filter, which means that the intensity of each pixel is
    replaced by a weighted average of intensity values from nearby pixels. The weights are computed
    using a Gaussian function of the spatial distance between pixels (the 'd' variable in the code
    below) and a Gaussian function of the intensity difference between pixels (the 'r' variable in
    the code below). The spatial Gaussian function is the same as the one used in the Gaussian blur.
    The intensity Gaussian is defined by the standard deviation of the intensity values in the
    neighborhood of the pixel (the 'sigma_r' variable in the code below). The 'sigma_r' variable is
    usually set to a small value, such as 0.1. The 'sigma_d' variable is the standard deviation of the
    spatial Gaussian function. It is usually set to a value slightly larger than the radius of the
    neighborhood of the pixel. The 'sigma_d' variable is usually set to a value slightly larger than
    the radius of the neighborhood of the pixel. The 'sigma_d' variable is usually set to a value
    slightly larger than the radius of the neighborhood of the pixel.
    
use: bilateral(<SAMPLER_TYPE> texture, <float2> st, <float2> duv)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERAL_AMOUNT
    - BILATERAL_TYPE
    - BILATERAL_SAMPLER_FNC
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
            float weight = (k2 / kernelSize2f) * gaussian(kernelSizef, float3(dx, dy, dl));
            accumColor += weight * t;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
