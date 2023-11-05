#include "aabb.cuh"
#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: return center of a AABB
use: <float3> centroid(<AABB> box) 
*/

#ifndef FNC_AABB_CENTROID
#define FNC_AABB_CENTROID
inline __host__ __device__ float3 centroid(const AABB& _box) { return (_box.min + _box.max) * 0.5f; }
#endif