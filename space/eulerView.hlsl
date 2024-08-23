/*
contributors:  Shadi El Hajj
description: Create a view matrix from camera position and camera rotation (euler angles)
use: <float4x4> eulerView(in <float3> position, in <float3> euler)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#include "../math/rotate3dX.hlsl"
#include "../math/rotate3dY.hlsl"
#include "../math/rotate3dZ.hlsl"
#include "translate.hlsl"

#ifndef FNC_EULERVIEW
#define FNC_EULERVIEW

 float4x4 eulerView(float3 position, float3 euler) {
    float3x3 rotZ = rotate3dZ(euler.z);
    float3x3 rotX = rotate3dX(euler.x);
    float3x3 rotY = rotate3dY(euler.y);
    float3x3 identity = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    float3x3 rotation = mul(rotY, mul(rotX, mul(rotZ, identity)));
    return translate(rotation, position);
}

#endif
