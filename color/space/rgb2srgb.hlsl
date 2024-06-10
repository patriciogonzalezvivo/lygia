/*
contributors: Patricio Gonzalez Vivo
description: Converts a linear RGB color to sRGB color space.
use: <float|float3\float4> rgb2srgb(<float|float3|float4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RGB2SRGB_EPSILON
#define RGB2SRGB_EPSILON 0.00000001
#endif

#ifndef FNC_RGB2SRGB
#define FNC_RGB2SRGB
// 1.0 / 2.4 = 0.4166666666666667 
float  rgb2srgb(float c) {      return (c < 0.0031308) ? c * 12.92 : 1.055 * pow(c, 0.4166666666666667) - 0.055; }
float3 rgb2srgb(float3 rgb) {   return saturate(float3( rgb2srgb(rgb[0] - RGB2SRGB_EPSILON), 
                                                        rgb2srgb(rgb[1] - RGB2SRGB_EPSILON), 
                                                        rgb2srgb(rgb[2] - RGB2SRGB_EPSILON))); }
float4 rgb2srgb(float4 rgb) {   return float4(rgb2srgb(rgb.rgb), rgb.a);}
#endif
