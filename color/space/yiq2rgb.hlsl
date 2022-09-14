/*
original_author: Patricio Gonzalez Vivo
description: pass a color in YIQ and get RGB color. From https://en.wikipedia.org/wiki/YIQ
use: yiq2rgb(<float3|float4> color)
*/

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB

const float3x3 yiq2rgb_mat = float3x3(
    1.,     1.,     1.,
    .956,  -.272, -1.106,
    .621,  -.647,  1.703
);

float3 yiq2rgb(in float3 yiq) {
    return mul(yiq2rgb_mat, yiq);
}

float4 yiq2rgb(in float4 yiq) {
    return float4(yiq2rgb(yiq.rgb), yiq.a);
}
#endif
