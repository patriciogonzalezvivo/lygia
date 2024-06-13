
/*
contributors: Johan Ismael
description: Similar to step but for an interval instead of a threshold. Returns 1 is x is between left and right, 0 otherwise
use: <float> within(<float> minVal, <float|float2|float3|float4> maxVal, <float|float2|float3|float4> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WITHIN
#define FNC_WITHIN
float within(in float minVal, in float maxVal, in float x) {
    return step(minVal, x) * (1. - step(maxVal, x));
}

float within(in float2 minVal, in float2 maxVal, in float2 x) {
    float2 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y;
}

float within(in float3 minVal, in float3 maxVal, in float3 x) {
    float3 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y * rta.z;
}

float within(in float4 minVal, in float4 maxVal, in float4 x) {
    float4 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y * rta.z * rta.w;
}
#endif