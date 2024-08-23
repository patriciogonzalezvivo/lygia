/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <float4x4> translate(in <float3x3> matrix, in <float3> tranaslation)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_TRANSLATE
#define FNC_TRANSLATE

 float4x4 translate(float3x3 m, float3 translation) {
    return float4x4(
        float4(m._m00_m01_m02, translation.x),
        float4(m._m10_m11_m12, translation.y),
        float4(m._m20_m21_m22, translation.z),
        float4(0.0, 0.0, 0.0, 1.0));
}

#endif
