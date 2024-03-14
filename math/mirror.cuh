#include "mod.cuh"
#include "frac.cuh"
#include "operations.cuh"

/*
contributors: Ian Heisters
description: Transforms the input signal into a triangle wave. For instance, if x goes between 0 and 2, the returned value will go from 0 to 1, and then 1 to 0 in a triangle shape.
use: <float|float2> mirror(<float|float2> x)
*/

#ifndef FNC_MIRROR
#define FNC_MIRROR
inline __device__ __host__ float mirror(float x) {
    float f = frac(x);
    float m = floor(mod(x, 2.0f));
    float fm = f * m;
    return f + m - fm * 2.;
}

inline __device__ __host__ float2 mirror(float2 xy) {
    float2 f = frac(xy);
    float2 m = floor(mod(xy, 2.0f));
    float2 fm = f * m;
    return f + m - fm * 2.0f;
}
#endif
