/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a LCh to Lab color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <float3|float4> lab2rgb(<float3|float4> color)
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