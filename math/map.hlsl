/*
contributors: Johan Ismael
description: Map a value between one range to another.
use: <float|float2|float3|float4> map(<float|float2|float3|float4> value, <float|float2|float3|float4> inMin, <float|float2|float3|float4> inMax, <float|float2|float3|float4> outMin, <float|float2|float3|float4> outMax)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MAP
#define FNC_MAP

float map( float value, float inMin, float inMax ) { return (value-inMin)/(inMax-inMin); }
float2 map( float2 value, float2 inMin, float2 inMax ) { return (value-inMin)/(inMax-inMin); }
float3 map( float3 value, float3 inMin, float3 inMax ) { return (value-inMin)/(inMax-inMin); }
float4 map( float4 value, float4 inMin, float4 inMax ) { return (value-inMin)/(inMax-inMin); }

float map(in float value, in float inMin, in float inMax, in float outMin, in float outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float2 map(in float2 value, in float2 inMin, in float2 inMax, in float2 outMin, in float2 outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float3 map(in float3 value, in float3 inMin, in float3 inMax, in float3 outMin, in float3 outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float4 map(in float4 value, in float4 inMin, in float4 inMax, in float4 outMin, in float4 outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

#endif
