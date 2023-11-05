#include "triangle.cuh"
#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: returns the centroid of a triangle
use: <float3> centroid(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_CENTROID
#define FNC_TRIANGLE_CENTROID

inline __host__ __device__ float3 centroid(const Triangle& _tri) { return (_tri.a + _tri.b + _tri.c) * 0.3333333333333f; }

#endif