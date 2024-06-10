#include "../math/rotate4dY.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<float3> pos, float radian [, float4 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATEY
#define FNC_ROTATEY
float3 rotateY(in float3 pos, in float radian, in float3 center) {
    return mul(rotate4dY(radian), float4((pos - center), 1.)).xyz + center;
}

float3 rotateY(in float3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateY(pos, radian, CENTER_3D);
    #else
    return rotateY(pos, radian, float3(0.0, 0.0, 0.0));
    #endif
}
#endif
