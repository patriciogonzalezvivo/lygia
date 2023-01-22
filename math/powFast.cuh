#include "operations.cuh"

/*
original_author: Patricio Gonzalez Vivo
description: fast approximation to pow()
use: powFast(<float> x, <float> exp)
*/

#ifndef FNC_POWFAST
#define FNC_POWFAST

inline __host__ __device__ float powFast(float a, float b) { return a / ((1.0f - b) * a + b); }

#endif