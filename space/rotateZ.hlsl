#include "../math/rotate4dZ.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateZ(<float3|float4> pos, float radian [, float3 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATEZ
#define FNC_ROTATEZ
float3 rotateZ(in float3 pos, in float radian, in float3 center) {
    return mul(rotate4dZ(radian), float4(pos - center, 0.) ).xyz + center;
}

float3 rotateZ(in float3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateZ(pos, radian, CENTER_3D);
    #else
    return rotateZ(pos, radian, float3(0.0, 0.0, 0.0));
    #endif
}
#endif
