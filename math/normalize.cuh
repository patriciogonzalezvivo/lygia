#include "dot.cuh"
#include "operations.cuh"

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_NORMALIZE
#define FNC_NORMALIZE

#ifndef __CUDACC__
#include <math.h>
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

inline __host__ __device__ float2 normalize(float2 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float3 normalize(float3 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float4 normalize(float4 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

#endif