#include "triangle.cuh"
#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the centroid of a triangle
use: <float3> centroid(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TRIANGLE_CENTROID
#define FNC_TRIANGLE_CENTROID

inline __host__ __device__ float3 centroid(const Triangle& _tri) { return (_tri.a + _tri.b + _tri.c) * 0.3333333333333f; }

#endif