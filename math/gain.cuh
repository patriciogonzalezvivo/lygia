#include "pow.cuh"

/*
contributors: Inigo Quiles
description: | 
    Remapping the unit interval into the unit interval by expanding the sides and compressing the center, and keeping 1/2 mapped to 1/2, that can be done with the gain() function. From https://iquilezles.org/articles/functions/
use: <float> gain(<float> x, <float> k)
*/

#ifndef FNC_GAIN
#define FNC_GAIN
inline __host__ __device__ float gain(float x, float k) {
    const float a = 0.5f * pow(2.0f * ((x < 0.5f)? x : 1.0f - x), k);
    return (x < 0.5f)? a : 1.0f - a;
}
#endif