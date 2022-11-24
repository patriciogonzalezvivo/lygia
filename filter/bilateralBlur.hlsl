#include "../color/space/rgb2luma.hlsl"
#include "../sample.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: TODO
use: bilateralBlur(<sampler2D> texture, <float2> st, <float2> duv)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERALBLUR_AMOUNT
    - BILATERALBLUR_TYPE
    - BILATERALBLUR_SAMPLER_FNC
*/

#ifndef BILATERALBLUR_AMOUNT
#define BILATERALBLUR_AMOUNT bilateralBlur13
#endif

#ifndef BILATERALBLUR_TYPE
#define BILATERALBLUR_TYPE float4
#endif

#ifndef BILATERALBLUR_SAMPLER_FNC
#define BILATERALBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef BILATERALBLUR_LUMA
#define BILATERALBLUR_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#include "bilateralBlur/2D.hlsl"

#ifndef FNC_BILATERALFILTER
#define FNC_BILATERALFILTER
BILATERALBLUR_TYPE bilateralBlur(in sampler2D tex, in float2 st, in float2 offset, const int kernelSize) {
    return bilateralBlur2D(tex, st, offset, kernelSize);
}

BILATERALBLUR_TYPE bilateralBlur13(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateralBlur(tex, st, offset, 7);
}

BILATERALBLUR_TYPE bilateralBlur9(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateralBlur(tex, st, offset, 5);
}

BILATERALBLUR_TYPE bilateralBlur5(in sampler2D tex, in float2 st, in float2 offset) {
    return bilateralBlur(tex, st, offset, 3);
}

BILATERALBLUR_TYPE bilateralBlur(in sampler2D tex, in float2 st, in float2 offset) {
    return BILATERALBLUR_AMOUNT(tex, st, offset);
}
#endif
