#include "aabb.cuh"

#include "../../math/min.cuh"
#include "../../math/max.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: compute the intersection of two AABBs
use: <float2> intersect(<AABB> box, <float3> rayOrigin, <float3> rayDir)
*/

#ifndef FNC_AABB_INTERSECTION
#define FNC_AABB_INTERSECTION

inline __host__ __device__ AABB intersection(const AABB& _a, const AABB& _b) {
    AABB rta;
    rta.min = max(_a.min, _b.min);
    rta.max = min(_a.max, _b.max);
    return rta;
}

#endif