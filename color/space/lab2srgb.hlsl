#include "lab2xyz.hlsl"
#include "xyz2srgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lab color to RGB color space. https://en.wikipedia.org/wiki/CIELAB_color_space
use: <float3|float4> lab2srgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LAB2SRGB
#define FNC_LAB2SRGB
float3 lab2srgb(const in float3 lab) {
    return xyz2srgb( lab2xyz( lab ) );
    // return xyz2srgb( lab2xyz( float3( 100.0 * lab.x,
    //                                 2.0 * 127.0 * (lab.y - 0.5),
    //                                 2.0 * 127.0 * (lab.z - 0.5)) ) );
}
float4 lab2srgb(const in float4 lab) { return float4(lab2srgb(lab.rgb), lab.a); }
#endif