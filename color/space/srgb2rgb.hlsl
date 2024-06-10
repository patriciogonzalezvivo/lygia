/*
contributors: Patricio Gonzalez Vivo
description: sRGB to linear RGB conversion.
use: <float|float3\float4> srgb2rgb(<float|float3|float4> srgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SRGB2RGB_EPSILON
#define SRGB2RGB_EPSILON 1e-10
#endif

#ifndef FNC_SRGB2RGB
#define FNC_SRGB2RGB
// 1. / 12.92 = 0.0773993808
// 1.0 / (1.0 + 0.055) = 0.947867298578199
float srgb2rgb(float c) {       return (c < 0.04045) ? c * 0.0773993808 : pow((c + 0.055) * 0.947867298578199, 2.4); }
float3 srgb2rgb(float3 srgb) {  return float3(  srgb2rgb(srgb[0] + SRGB2RGB_EPSILON), 
                                                srgb2rgb(srgb[1] + SRGB2RGB_EPSILON), 
                                                srgb2rgb(srgb[2] + SRGB2RGB_EPSILON)); }
float4 srgb2rgb(float4 srgb) {  return float4(srgb2rgb(srgb.rgb), srgb.a); }
#endif
