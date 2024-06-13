/*
contributors: Patricio Gonzalez Vivo
description: "Converts a LCh to Lab color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
use: <float3|float4> lab2rgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LAB2LCH
#define FNC_LAB2LCH
float3 lab2lch(float3 lab) {
    return float3(
        lab.x,
        sqrt(dot(lab.yz, lab.yz)),
        atan(lab.z, lab.y) * 57.2957795131
    );
}
float4 lab2lch(float4 lab) { return float4(lab2lch(lab.xyz), lab.a); }
#endif