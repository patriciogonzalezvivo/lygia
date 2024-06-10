#include "operations.cuh"
#include "step.cuh"

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
inline __host__ __device__ float within(float minVal, float maxVal, float x) {
    return step(minVal, x) * (1.0f - step(maxVal, x));
}

inline __host__ __device__ float within(const float2& minVal, const float2& maxVal, const float2& x) {
    float2 rta = step(minVal, x) * (1.0f - step(maxVal, x));
    return rta.x * rta.y;
}

inline __host__ __device__ float within(const float3& minVal, const float3& maxVal, const float3& x) {
    float3 rta = step(minVal, x) * (1.0f - step(maxVal, x));
    return rta.x * rta.y * rta.z;
}

inline __host__ __device__ float within(const float4& minVal, const float4& maxVal, const float4& x) {
    float4 rta = step(minVal, x) * (1.0f - step(maxVal, x));
    return rta.x * rta.y * rta.z * rta.w;
}
#endif