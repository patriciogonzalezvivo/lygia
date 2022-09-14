/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YUB
use: rgb2yuv(<float3|float4> color)
*/

#ifndef FNC_RGB2YUV
#define FNC_RGB2YUV

#ifdef YUV_SDTV
const float3x3 rgb2yuv_mat = float3x3(
    .299, -.14713,  .615,
    .587, -.28886, -.51499,
    .114,  .436,   -.10001
);
#else
const float3x3 rgb2yuv_mat = float3x3(
    .2126,  -.09991, .615,
    .7152,  -.33609,-.55861,
    .0722,   .426,  -.05639
);
#endif

float3 rgb2yuv(in float3 rgb) {
    return mul(rgb2yuv_mat, rgb);
}

float4 rgb2yuv(in float4 rgb) {
    return float4(rgb2yuv(rgb.rgb),rgb.a);
}
#endif
