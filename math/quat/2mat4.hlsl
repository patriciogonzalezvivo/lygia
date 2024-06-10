#include "type.hlsl"
#include "2mat3.hlsl"
#include "../toMat4.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns a rotation 4x4 matrix
use: <mat4> quat2mat4(<QUAT> Q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef FNC_QUAT2MAT4
#define FNC_QUAT2MAT4
float4x4 quat2mat4(QUAT q) { return toMat4(quat2mat3(q)); }
#endif