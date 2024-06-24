// #include "../math/saturate.cuh"
#include "../math/mix.cuh"
#include "../math/clamp.cuh"

/*
contributors:  Inigo Quiles
description: Union operation of two SDFs 
use: <float> opUnion( in <float|float4> d1, in <float|float4> d2 [, <float> smooth_factor] ) 
*/

#ifndef FNC_OPUNION
#define FNC_OPUNION

inline __host__ __device__ float opUnion( float d1, float d2 ) { return min(d1, d2); }
inline __host__ __device__ float4 opUnion( float4 d1, float4 d2 ) { return (d1.w < d2.w) ? d1 : d2; }

// Soft union
inline __host__ __device__ float opUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5f + 0.5f * (d2-d1)/k, 0.0f, 1.0f );
    return mix( d2, d1, h ) - k * h * (1.0f-h); 
}

inline __host__ __device__ float4 opUnion( float4 d1, float4 d2, float k ) {
    float h = clamp( 0.5f + 0.5f * (d2.w - d1.w)/k, 0.0f, 1.0f );
    float4 result = mix( d2, d1, h ); 
    result.w -= k * h * (1.0f-h);
    return result;
}

#endif