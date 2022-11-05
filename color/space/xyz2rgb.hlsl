#include "linear2gamma.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a XYZ color to RGB color space.
use: xyz2rgb(<float3|float4> color)
*/

#ifndef FNC_XYZ2RGB
#define FNC_XYZ2RGB
float3 xyz2rgb(in float3 c) {
    const float3x3 mat = float3x3(  3.2404542, -0.9692660,  0.0556434,
                                    -1.5371585,  1.8760108, -0.2040259,
                                    -0.4985314,  0.0415560,  1.0572252);

    float3 v = mul(mat, c / 100.0);
    float3 c0 = (1.055 * linear2gamma(v)) - 0.055;
    float3 c1 = 12.92 * v;
    float3 r = lerp(c0, c1, step(v, float3(0.0031308)));
    return r;
}

float4 xyz2rgb(in float4 xyz) { return float4(xyz2rgb(xyz.rgb), xyz.a); }
#endif