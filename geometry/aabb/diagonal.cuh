
#include "aabb.cuh"
#include "../../math/abs.cuh"
#include "../../math/operations.cuh"

/*
contributors: Patrincio Gonzalez Vivo
description: return the diagonal vector of a AABB
use: <float> diagonal(<AABB> box ) 
*/

#ifndef FNC_AABB_DIAGONAL
#define FNC_AABB_DIAGONAL

inline __host__ __device__ float3 diagonal(const AABB& box) { return abs(box.max - box.min); }

#endif