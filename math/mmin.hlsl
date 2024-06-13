/*
contributors: Patricio Gonzalez Vivo
description: extend HLSL min function to add more arguments
use:
    - <float> mmin(<float> A, <float> B, <float> C[, <float> D])
    - <float2|float3|float4> mmin(<float2|float3|float4> A)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MMIN
#define FNC_MMIN

float mmin(in float a, in float b) { return min(a, b); }
float mmin(in float a, in float b, in float c) { return min(a, min(b, c)); }
float mmin(in float a, in float b, in float c, in float d) { return min(min(a,b), min(c, d)); }

float mmin(const float2 v) { return min(v.x, v.y); }
float mmin(const float3 v) { return mmin(v.x, v.y, v.z); }
float mmin(const float4 v) { return mmin(v.x, v.y, v.z, v.w); }

#endif
