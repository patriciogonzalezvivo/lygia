/*
contributors: Patricio Gonzalez Vivo
description: Flip Y axis
use: <float2|float3|float4> flipY(<float2|float3|float4> st)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FLIPY
#define FNC_FLIPY
float2 flipY(in float2 st) { return float2(st.x, 1. - st.y);}
float3 flipY(in float3 st) { return float3(st.x, 1. - st.y, st.z);}
float4 flipY(in float4 st) { return float4(st.x, 1. - st.y, st.z, st.w);}
#endif
