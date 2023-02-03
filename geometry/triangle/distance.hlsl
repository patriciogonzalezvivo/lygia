#include "distanceSq.hlsl"

/*
original_author: Inigo Quiles
description: returns the closest distance to the surface of a triangle
use: <float3> closestDistance(<Triangle> tri, <float3> _pos) 
*/

#ifndef FNC_TRIANGLE_CLOSEST_DISTANCE
#define FNC_TRIANGLE_CLOSEST_DISTANCE

float distance(Triangle _tri, float3 _pos) { return sqrt( distance(_tri, _pos) ); }

#endif