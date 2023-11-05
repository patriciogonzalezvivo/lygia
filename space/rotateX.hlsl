#include "../math/rotate4dX.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateX(<float3|float4> pos, float radian [, float3 center])
options:
    - CENTER_3D
*/

#ifndef FNC_ROTATEX
#define FNC_ROTATEX
float3 rotateX(in float3 pos, in float radian, in float3 center) {
    return mul(rotate4dX(radian), float4(pos - center, 1.)).xyz + center;
}

float3 rotateX(in float3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateX(pos, radian, CENTER_3D);
    #else
    return rotateX(pos, radian, float3(0.0, 0.0, 0.0));
    #endif
}
#endif
