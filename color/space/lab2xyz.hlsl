/*
original_author: Patricio Gonzalez Vivo
description: Converts a Lab color to XYZ color space.
use: rgb2xyz(<float3|vec4> color)
*/

#ifndef FNC_LAB2XYZ
#define FNC_LAB2XYZ
float3 lab2xyz(in float3 c) {
    float3 f = float3(0.0, 0.0, 0.0);
    f.y = (c.x + 16.0) / 116.0;
    f.x = c.y / 500.0 + f.y;
    f.z = f.y - c.z / 200.0;
    float3 c0 = f * f * f;
    float3 c1 = (f - 16.0 / 116.0) / 7.787;
    return float3(95.047, 100.000, 108.883) * lerp(c0, c1, step(f, float3(0.206897, 0.206897, 0.206897)));
}

vec4 lab2xyz(in vec4 c) { return vec4(lab2xyz(c.xyz), c.w); }
#endif