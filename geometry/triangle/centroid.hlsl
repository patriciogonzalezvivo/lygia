#include "triangle.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: returns the centroid of a triangle
use: <float3> centroid(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_CENTROID
#define FNC_TRIANGLE_CENTROID

float3 centroid(Triangle _tri) { return (_tri.a + _tri.b + _tri.c) * 0.3333333333333f; }

#endif