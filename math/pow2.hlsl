/*
contributors: Patricio Gonzalez Vivo
description: power of 2
use: <float|float2|float3|float4> pow2(<float|float2|float3|float4> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_POW2
#define FNC_POW2

float  pow2(in float x) { return x * x; }
float2 pow2(in float2 x) { return x * x; }
float3 pow2(in float3 x) { return x * x; }
float4 pow2(in float4 x) { return x * x; }

#endif
