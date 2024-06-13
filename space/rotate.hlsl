#include "../math/rotate2d.hlsl"
#include "../math/rotate4d.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian radians
use: rotate(<float3|float2> st, float radians [, float2 center])
options:
    - CENTER_2D
    - CENTER_3D
    - CENTER_4D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE
#define FNC_ROTATE
float2 rotate(in float2 st, in float radians, in float2 center) {
    return mul(rotate2d(radians), (st - center)) + center;
}

float2 rotate(in float2 st, in float radians) {
    #ifdef CENTER_2D
    return rotate(st, radians, CENTER_2D);
    #else
    return rotate(st, radians, float2(0.5, 0.5));
    #endif
}

float2 rotate(float2 st, float2 x_axis) {
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #endif
    float2 rta = float2( dot(st, float2(-x_axis.y, x_axis.x)), dot(st, x_axis) );
    #ifdef CENTER_2D
    rta += CENTER_2D;
    #endif
    return rta;
}

float3 rotate(in float3 xyz, in float radians, in float3 axis, in float3 center) {
    return mul(rotate4d(axis, radians), float4(xyz - center, 1.)).xyz + center;
}

float3 rotate(in float3 xyz, in float radians, in float3 axis) {
    #ifdef CENTER_3D
    return rotate(xyz, radians, axis, CENTER_3D);
    #else
    return rotate(xyz, radians, axis, float3(0.0, 0.0, 0.0));
    #endif
}

float4 rotate(in float4 xyzw, in float radians, in float3 axis, in float4 center) {
    return mul(rotate4d(axis, radians), (xyzw - center)) + center;
}

float4 rotate(in float4 xyzw, in float radians, in float3 axis) {
    #ifdef CENTER_4D
    return rotate(xyzw, radians, axis, CENTER_4D);
    #else
    return rotate(xyzw, radians, axis, float4(0.0, 0.0, 0.0, 0.0));
    #endif
}
#endif
