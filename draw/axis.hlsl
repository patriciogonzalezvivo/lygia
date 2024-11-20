
#include "line.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draw the three axis (X,Y,Z) of a 3D object projected in 2D. The thickness of the axis can be adjusted.
use: <float4> axis(<float2> st, <float4x4> M, <float3> pos, <float> thickness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AXIS
#define FNC_AXIS

float4 axis(in float2 st, float4x4 M, float3 pos, float thickness) {
    float4 rta = float4(0.0);

    float4 center = mult(M, float4(pos, 1.0));
    center.xy /= center.w;
    center.xy = (center.xy * 0.5 + 0.5);

    float4 axis[3];
    axis[0] = float4(1.0, 0.0, 0.0, 1.0);
    axis[1] = float4(0.0, 1.0, 0.0, 1.0);
    axis[2] = float4(0.0, 0.0, 1.0, 1.0);

    for (int i = 0; i < 3; i++) {
        #ifdef DEBUG_FLIPPED_SPACE
        float4 a = M * (float4(pos - axis[i].xyz, 1.0));
        #else
        float4 a = M * (float4(pos + axis[i].xyz, 1.0));
        #endif
        a.xy /= a.w;
        a.xy = (a.xy * 0.5 + 0.5);
        rta += axis[i] * line(st, center.xy, a.xy, thickness);
    } 
     
    return rta;
}

#endif