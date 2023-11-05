#include "diagonal.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: square a AABB using the longest side
use: <void> square(<AABB> box) 
*/

#ifndef FNC_AABB_SQUARE
#define FNC_AABB_SQUARE

inline __host__ __device__ void square(AABB& _box) {
    float3 diag     = diagonal(_box) * 0.5f;
    float3 cntr     = _box.min + diag;
    float  mmax     = max( abs(diag.x), max( abs(diag.y), abs(diag.z) ) );
    _box.max         = cntr + mmax;
    _box.min         = cntr - mmax;
}

#endif