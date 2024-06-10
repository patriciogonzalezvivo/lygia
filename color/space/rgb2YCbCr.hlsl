/*
contributors: Patricio Gonzalez Vivo
description: Convert RGB to YCbCr according to https://en.wikipedia.org/wiki/YCbCr
use: rgb2YCbCr(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2YCBCR
#define FNC_RGB2YCBCR
float3 rgb2YCbCr(in float3 rgb){
    float y = dot(rgb, float3(.299, .587, .114));
    float cb = .5 + dot(rgb, float3(-.168736, -.331264, .5));
    float cr = .5 + dot(rgb, float3(.5, -.418688, -.081312));
    return float3(y, cb, cr);
}

float4 rgb2YCbCr(in float4 rgb) {
    return float4(rgb2YCbCr(rgb.rgb),rgb.a);
}
#endif