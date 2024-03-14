/*
contributors: Inigo Quiles
description: cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> cubic(<float|float2|float3|float4> value [, <float> in, <float> out]);
*/

#ifndef FNC_CUBIC
#define FNC_CUBIC 
float   cubic(const in float v)   { return v*v*(3.0-2.0*v); }
float2  cubic(const in float2 v)  { return v*v*(3.0-2.0*v); }
float3  cubic(const in float3 v)  { return v*v*(3.0-2.0*v); }
float4  cubic(const in float4 v)  { return v*v*(3.0-2.0*v); }

float cubic(const in float value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    float value2 = value * value;
    float value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

float2 cubic(const in float2 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    float2 value2 = value * value;
    float2 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

float3 cubic(const in float3 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    float3 value2 = value * value;
    float3 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

float4 cubic(const in float4 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    float4 value2 = value * value;
    float4 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}
#endif