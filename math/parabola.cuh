#include "pow.cuh"

/*
contributors: Inigo Quiles
description: |
    A nice choice to remap the 0..1 interval into 0..1, such that the corners are mapped to 0 and the center to 1. You can then rise the parabolar to a power k to control its shape. From https://iquilezles.org/articles/functions/
use: <float> parabola(<float> x, <float> k)
*/

#ifndef FNC_PARABOLA
#define FNC_PARABOLA

inline __device__ __host__ float parabola( float x, float k ) { return pow( 4.0f * x * (1.0f - x), k ); }

#endif