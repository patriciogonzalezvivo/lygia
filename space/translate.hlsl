/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <float4x4> translate(in <float3x3> matrix, in <float3> tranaslation)
*/

#ifndef FNC_TRANSLATE
#define FNC_TRANSLATE

 float4x4 translate(float3x3 m, float3 translation) {
    float4x4 m4 = float4x4(m);
    m4._m03_m13_m23_m33 = float4(translation, 1.0);
    return m4;
}

#endif
