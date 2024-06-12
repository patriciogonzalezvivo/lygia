/*
contributors: Patricio Gonzalez Vivo
description: An implementation of mod that matches the GLSL mod. Note that HLSL's fmod is different.
use: <float|float2|float3|float4> mod(<float|float2|float3|float4> value, <float|float2|float3|float4> modulus)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MOD
#define FNC_MOD

float mod(in float x, in float y) { return x - y * floor(x / y); }
float2 mod(in float2 x, in float2 y) { return x - y * floor(x / y); }
float3 mod(in float3 x, in float3 y) { return x - y * floor(x / y); }
float4 mod(in float4 x, in float4 y) { return x - y * floor(x / y); }

#endif