
#include "diagonal.cuh"

/*
original_author: Patricio Gonzalez Vivo
description: return center of a AABB
use: <float3> AABBcenter(<AABB> box) 
*/

#ifndef FNC_AABB_CENTER
#define FNC_AABB_CENTER

inline __host__ __device__ float3 AABBcenter(const AABB& box) { return diagonal(box) * 0.5f; }

#endif