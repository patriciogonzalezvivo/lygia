/*
original_author: Patricio Gonzalez Vivo
description: gaussian coeficient
use: <float4|float3|float2|float> gaussian(<float> sigma, <float4|float3|float2|float> d)
*/

#ifndef FNC_GAUSSIAN
#define FNC_GAUSSIAN

inline __host__ __device__ float gaussian(float sigma, float d) { return exp(-(d*d) / (2.0 * sigma*sigma)); }
inline __host__ __device__ float gaussian(float sigma, float2 d) { return exp(-( d.x*d.x + d.y*d.y) / (2.0 * sigma*sigma)); }
inline __host__ __device__ float gaussian(float sigma, float3 d) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z ) / (2.0 * sigma*sigma)); }
inline __host__ __device__ float gaussian(float sigma, float4 d) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w ) / (2.0 * sigma*sigma)); }

#endif