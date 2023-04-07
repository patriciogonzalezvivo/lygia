/*
original_author: Patricio Gonzalez Vivo  
description: Converts a linear RGB color to sRGB color space.
use: <float|float3\float4> rgb2srgb(<float|float3|float4> srgb)
*/

#ifndef SRGB_EPSILON 
#define SRGB_EPSILON 0.00000001
#endif

#ifndef FNC_RGB2SRGB
#define FNC_RGB2SRGB

float rgb2srgb(float channel) {
    return (channel < 0.0031308) ? channel * 12.92 : 1.055 * pow(channel, 0.4166666666666667) - 0.055;
}

float3 rgb2srgb(float3 rgb) {
    return saturate(float3(rgb2srgb(rgb[0] - SRGB_EPSILON), rgb2srgb(rgb[1] - SRGB_EPSILON), rgb2srgb(rgb[2] - SRGB_EPSILON)));
}

float4 rgb2srgb(float4 rgb) {
    return float4(rgb2srgb(rgb.rgb), rgb.a);
}

#endif