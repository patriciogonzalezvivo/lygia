#include "operations.cuh"

/*
contributors: Inigo Quiles
description: cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> cubic(<float|float2|float3|float4> value [, <float> in, <float> out]);
*/

#ifndef FNC_CUBIC
#define FNC_CUBIC 
inline __host__ __device__ float   cubic(float v)   { return v * v * (3.0f - 2.0f * v); }
inline __host__ __device__ float2  cubic(float2 v)  { return v * v * (3.0f - 2.0f * v); }
inline __host__ __device__ float3  cubic(float3 v)  { return v * v * (3.0f - 2.0f * v); }
inline __host__ __device__ float4  cubic(float4 v)  { return v * v * (3.0f - 2.0f * v); }

inline __host__ __device__ float cubic(float value, float slope0, float slope1) {
    float a = slope0 + slope1 - 2.0f;
    float b = -2.0f * slope0 - slope1 + 3.0f;
    float c = slope0;
    float value2 = value * value;
    float value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

inline __host__ __device__ float2 cubic(float2 value, float slope0, float slope1) {
    float a = slope0 + slope1 - 2.0f;
    float b = -2.0f * slope0 - slope1 + 3.0f;
    float c = slope0;
    float2 value2 = value * value;
    float2 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

inline __host__ __device__ float3 cubic(float3 value, float slope0, float slope1) {
    float a = slope0 + slope1 - 2.0f;
    float b = -2.0f * slope0 - slope1 + 3.0f;
    float c = slope0;
    float3 value2 = value * value;
    float3 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

inline __host__ __device__ float4 cubic(float4 value, float slope0, float slope1) {
    float a = slope0 + slope1 - 2.0f;
    float b = -2.0f * slope0 - slope1 + 3.0f;
    float c = slope0;
    float4 value2 = value * value;
    float4 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}
#endif