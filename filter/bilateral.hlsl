#include "../color/space/rgb2luma.hlsl"
#include "../sample.hlsl"

/*
original_author: Patricio Gonzalez Vivo
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
    
use: bilateral(<sampler2D> texture, <float2> st, <float2> duv)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERAL_AMOUNT
    - BILATERAL_TYPE
    - BILATERAL_SAMPLER_FNC
*/

#ifndef BILATERAL_AMOUNT
#define BILATERAL_AMOUNT bilateral13
#endif

#ifndef BILATERAL_TYPE
#define BILATERAL_TYPE float4
#endif

#ifndef BILATERAL_SAMPLER_FNC
#define BILATERAL_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef BILATERAL_LUMA
#define BILATERAL_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#include "bilateral/2D.hlsl"

#ifndef FNC_BILATERALFILTER
#define FNC_BILATERALFILTER
BILATERAL_TYPE bilateral(in sampler2D tex, in float2 st, in float2 offset, const int kernelSize) {
    return bilateral2D(tex, st, offset, kernelSize);
}

BILATERAL_TYPE bilateral13(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateral(tex, st, offset, 7);
}

BILATERAL_TYPE bilateral9(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateral(tex, st, offset, 5);
}

BILATERAL_TYPE bilateral5(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateral(tex, st, offset, 3);
}

BILATERAL_TYPE bilateral(in sampler2D tex, in float2 st, in float2 offset) {
    return BILATERAL_AMOUNT(tex, st, offset);
}
#endif
