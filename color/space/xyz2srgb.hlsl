#include "rgb2srgb.hlsl"

/*
original_author: Ronald van Wijnen (@OneDayOfCrypto)
description: Converts a XYZ color to sRGB color space.
use: xyz2rgb(<float3|float4> color)
*/

#ifndef FNC_XYZ2SRGB
#define FNC_XYZ2SRGB
float3 xyz2srgb(float3 xyz) {
    float3x3 D65_XYZ_RGB;
    D65_XYZ_RGB[0] = float3( 3.24306333, -1.53837619, -0.49893282);
    D65_XYZ_RGB[1] = float3(-0.96896309,  1.87542451,  0.04154303);
    D65_XYZ_RGB[2] = float3( 0.05568392, -0.20417438,  1.05799454);
    
    float r = dot(D65_XYZ_RGB[0], xyz);
    float g = dot(D65_XYZ_RGB[1], xyz);
    float b = dot(D65_XYZ_RGB[2], xyz);
    return rgb2srgb(float3(r, g, b));
}

float4 xyz2srgb(in float4 xyz) { return float4(xyz2srgb(xyz.rgb), xyz.a); }
#endif

