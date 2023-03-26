/*
original_author: Patricio Gonzalez Vivo
description: | 
    Convert from linear RGB to YIQ which was the followin range. 
    Using conversion matrices from FCC NTSC Standard (SMPTE C) https://en.wikipedia.org/wiki/YIQ
use: <float3|float4> rgb2yiq(<float3|float4> color)
*/

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ
const float3x3 rgb2yiq_mat = float3x3(
    0.300,  0.5900,  0.1100, 
    0.599, -0.2773, -0.3217, 
    0.213, -0.5251,  0.3121
);

float3 rgb2yiq(in float3 rgb) {
  return mul(rgb2yiq_mat, rgb);
}

float4 rgb2yiq(in float4 rgb) {
    return float4(rgb2yiq(rgb.rgb), rgb.a);
}
#endif
