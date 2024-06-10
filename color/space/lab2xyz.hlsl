/*
contributors: Patricio Gonzalez Vivo
description: Converts a Lab color to XYZ color space.
use: rgb2xyz(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CIE_WHITE
#ifdef CIE_D50
#define CIE_WHITE float3(0.96429567643, 1.0, 0.82510460251)
#else
// D65
#define CIE_WHITE float3(0.95045592705, 1.0, 1.08905775076)
#endif
#endif

#ifndef FNC_LAB2XYZ
#define FNC_LAB2XYZ

#ifndef FNC_LAB2XYZ
#define FNC_LAB2XYZ
float3 lab2xyz(const in float3 c) {
    float fy = ( c.x + 16.0 ) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return CIE_WHITE * 100.0 * float3(
        ( fx > 0.206897 ) ? fx * fx * fx : ( fx - 16.0 / 116.0 ) / 7.787,
        ( fy > 0.206897 ) ? fy * fy * fy : ( fy - 16.0 / 116.0 ) / 7.787,
        ( fz > 0.206897 ) ? fz * fz * fz : ( fz - 16.0 / 116.0 ) / 7.787
    );
}

float4 lab2xyz(in float4 c) { return float4(lab2xyz(c.xyz), c.w); }
#endif