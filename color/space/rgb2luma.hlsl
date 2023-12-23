/*
contributors: Patricio Gonzalez Vivo
description: |
    Get's the luminosity from linear RGB, based on Rec709 luminance (see https://en.wikipedia.org/wiki/Grayscale)
use: <float> rgb2luma(<float3|float4> rgb)
*/

#ifndef FNC_RGB2LUMA
#define FNC_RGB2LUMA
float rgb2luma(const in float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }
float rgb2luma(const in float4 rgb) { return rgb2luma(rgb.rgb); }
#endif
