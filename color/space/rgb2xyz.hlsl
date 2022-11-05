#include "gamma2linear.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB color to XYZ color space.
use: rgb2xyz(<float3|float4> color)
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
float3 rgb2xyz(in float3 c) {
    const float3x3 mat = float3x3(  0.4124564, 0.2126729, 0.0193339,
                                    0.3575761, 0.7151522, 0.1191920,
                                    0.1804375, 0.0721750, 0.9503041);
    float3 c0 = gamma2linear((c + 0.055) / 1.055);
    float3 c1 = c / 12.92;
    float3 tmp = lerp(c0, c1, step(c, float3(0.04045)));
    return mul(mat, 100.0 * tmp);
}

float4 rgb2xyz(in float4 rgb) { return float4(rgb2xyz(rgb.rgb),rgb.a); }
#endif