/*
contributors: Patricio Gonzalez Vivo
description: power of 5
use: <float|float2|float3|float4> pow5(<float|float2|float3|float4> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_POW5
#define FNC_POW5

float pow5(in float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float2 pow5(in float2 x) {
    float2 x2 = x * x;
    return x2 * x2 * x;
}

float3 pow5(in float3 x) {
    float3 x2 = x * x;
    return x2 * x2 * x;
}

float4 pow5(in float4 x) {
    float4 x2 = x * x;
    return x2 * x2 * x;
}

#endif
