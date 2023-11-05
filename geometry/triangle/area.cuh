#include "triangle.cuh"
#include "../../math/operations.cuh"
#include "../../math/cross.cuh"
#include "../../math/length.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: returns the area of a triangle
use: <float3> normal(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_AREA
#define FNC_TRIANGLE_AREA
inline __host__ __device__ float area(const Triangle& _tri) { return length( cross( _tri.b - _tri.a, _tri.c - _tri.a) ) * 0.5f; }
#endif