/*
original_author: Patricio Gonzalez Vivo
description: pass a color in YUB and get RGB color
use: yuv2rgb(<float3|float4> color)
*/

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB

#ifdef YUV_SDTV
const float3x3 yuv2rgb_mat = float3x3(
    1.,       1. ,      1.,
    0.,       -.39465,  2.03211,
    1.13983,  -.58060,  0.
);
#else
const float3x3 yuv2rgb_mat = float3x3(
    1.,       1. ,      1.,
    0.,       -.21482,  2.12798,
    1.28033,  -.38059,  0.
);
#endif

float3 yuv2rgb(in float3 yuv) {
    return mul(yuv2rgb_mat, yuv);
}

float4 yuv2rgb(in float4 yuv) {
    return float4(yuv2rgb(yuv.rgb), yuv.a);
}
#endif
