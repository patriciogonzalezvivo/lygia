/*
contributors: Patricio Gonzalez Vivo
description: |
    convert from xyY to XYZ
use: <float3|float4> xyY2xyz(<float3|float4> color)
*/

#ifndef FNC_XYY2XYZ
#define FNC_XYY2XYZ
float3 xyY2xyz(float3 xyY) {
    float Y = xyY.z;
    float f = 1.0/xyY.y;
    float x = Y * xyY.x * f;
    float z = Y * (1.0 - xyY.x - xyY.y) * f;
    return float3(x, Y, z);
}
float4 xyY2xyz(float4 xyY) { return float4(xyY2xyz(xyY.xyz), xyY.w); }
#endif