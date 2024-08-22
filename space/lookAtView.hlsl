/*
contributors:  Shadi El Hajj
description: Create a look-at view matrix
use: <float4x4> lookAtView(in <float3> position, in <float3> target, in <float3> up)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#include "lookAt.hlsl"
#include "translate.hlsl"

#ifndef FNC_LOOKATVIEW
#define FNC_LOOKATVIEW

float4x4 lookAtView( in float3 position, in float3 target, in float3 up ) {
    float3x3 m = lookAt(position, target, up);
    return translate(m, position);
}

float4x4 lookAtView( in float3 position, in float3 target, in float roll ) {
    float3x3 m = lookAt(position, target, roll);
    return translate(m, position);
}

float4x4 lookAtView( in float3 position, in float3 lookAt ) {
    return lookAtView(position, lookAt, float3(0.0, 1.0, 0.0));
}

#endif