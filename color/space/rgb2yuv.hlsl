/*
contributors: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YUB
use: rgb2yuv(<float3|float4> color)
*/

#ifndef MAT_RGB2YUV
#define MAT_RGB2YUV
#ifdef YUV_SDTV
const float3x3 RGB2YUV = float3x3(
    0.299, -0.14713,  0.615,
    0.587, -0.28886, -0.51499,
    0.114,  0.436,   -0.10001
);
#else
const float3x3 RGB2YUV = float3x3(
    0.2126,  -0.09991, 0.615,
    0.7152,  -0.33609,-0.55861,
    0.0722,   0.426,  -0.05639
);
#endif
#endif

#ifndef FNC_RGB2YUV
#define FNC_RGB2YUV
float3 rgb2yuv(in float3 rgb) { return mul(RGB2YUV, rgb); }
float4 rgb2yuv(in float4 rgb) { return float4(rgb2yuv(rgb.rgb),rgb.a); }
#endif
