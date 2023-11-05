#include "triangle.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: returns the normal of a triangle
use: <float3> getNormal(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_NORMAL
#define FNC_TRIANGLE_NORMAL

float3 normal(Triangle _tri) { return normalize( cross( _tri.b - _tri.a, _tri.c - _tri.a) ); }

#endif