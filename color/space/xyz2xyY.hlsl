/*
contributors: Patricio Gonzalez Vivo
description: Converts from XYZ to xyY space (Y is the luminance)
use: <float3|float4>  xyz2rgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYZ2XYY 
#define FNC_XYZ2XYY
float3 xyz2xyY(float3 xyz) {
    float Y = xyz.y;
    float f = 1.0 / (xyz.x + xyz.y + xyz.z);
    float x = xyz.x * f;
    float y = xyz.y * f;
    return float3(x, y, Y);
}
float4 xyz2xyY(float4 xyz) { return float4(xyz2xyY(xyz.xyz), xyz.w);}
#endif