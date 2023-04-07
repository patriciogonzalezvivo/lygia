/*
original_author: Patricio Gonzalez Vivo  
description: sRGB to linear RGB conversion.
use: <float|float3\float4> srgb2rgb(<float|float3|float4> srgb)
*/

#ifndef SRGB_EPSILON 
#define SRGB_EPSILON 0.00000001
#endif

#ifndef FNC_SRGB2RGB
#define FNC_SRGB2RGB

// 1. / 12.92 = 0.0773993808
// 1.0 / (1.0 + 0.055) = 0.947867298578199

float srgb2rgb(float channel) {
    return (channel < 0.04045) ? channel * 0.0773993808 : pow((channel + 0.055) * 0.947867298578199, 2.4);
}

float3 srgb2rgb(float3 srgb) {
    return float3(srgb2rgb(srgb[0] + SRGB_EPSILON), srgb2rgb(srgb[1] + SRGB_EPSILON), srgb2rgb(srgb[2] + SRGB_EPSILON));
}

float4 srgb2rgb(float4 srgb) {
    return float4(srgb2rgb(srgb.rgb), srgb.a);
}

#endif