#include "xyz2srgb.hlsl"
#include "xyY2xyz.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts from xyY to sRGB'
use: <float3|float4> xyY2srgb(<float3|float4> xyY)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYY2SRGB
#define FNC_XYY2SRGB
float3 xyY2srgb(const in float3 xyY) { return xyz2srgb(xyY2xyz(xyY));}
float4 xyY2srgb(const in float4 xyY) { return float4(xyY2srgb(xyY.xyz), xyY.w);}
#endif
