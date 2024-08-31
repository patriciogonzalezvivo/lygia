/*
contributors:  Shadi El Hajj
description: Tangent-Bitangent-Normal Matrix
use: float3x3 tbn(float3 t, float3 b, float3 n)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

#ifndef FNC_TBN
#define FNC_TBN

float3x3 tbn(float3 t, float3 b, float3 n) {
    float3x3 M;
    M._m00_m10_m20 = t;
    M._m01_m11_m21 = b;
    M._m02_m12_m22 = n;
    return M;
}

float3x3 tbn(float3 n, float3 up) {
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    return tbn(t, b, n);
}

#endif