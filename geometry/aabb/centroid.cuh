#include "aabb.cuh"
#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Return center of a AABB
use: <float3> centroid(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AABB_CENTROID
#define FNC_AABB_CENTROID
inline __host__ __device__ float3 centroid(const AABB& _box) { return (_box.min + _box.max) * 0.5f; }
#endif