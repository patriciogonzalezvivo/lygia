#include "triangle.cuh"
#include "../../math/cross.cuh"
#include "../../math/operations.cuh"
#include "../../math/normalize.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: returns the normal of a triangle
use: <float3> getNormal(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_NORMAL
#define FNC_TRIANGLE_NORMAL

inline __host__ __device__ float3 normal(const Triangle& _tri) { return normalize( cross( _tri.b - _tri.a, _tri.c - _tri.a) ); }

#endif