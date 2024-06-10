#include "clamp.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and 1
use: <float|vec2|vec3|vec4> saturation(<float|vec2|vec3|vec4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SATURATE
#define FNC_SATURATE

inline  __host__ __device__ float  saturate( float x ){ return clamp(x, 0.0f, 1.0f); }
inline  __host__ __device__ float2 saturate( const float2& x ){ return clamp(x, 0.0f, 1.0f); }
inline  __host__ __device__ float3 saturate( const float3& x ){ return clamp(x, 0.0f, 1.0f); }
inline  __host__ __device__ float4 saturate( const float4& x ){ return clamp(x, 0.0f, 1.0f); }

#endif