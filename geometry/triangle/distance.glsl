#include "distanceSq.hlsl"

/*
original_author: Inigo Quiles
description: returns the closest distance to the surface of a triangle
use: <vec3> closestDistance(<Triangle> tri, <vec3> _pos) 
*/

#ifndef FNC_TRIANGLE_CLOSEST_DISTANCE
#define FNC_TRIANGLE_CLOSEST_DISTANCE

float distance(Triangle _tri, vec3 _pos) { return sqrt( distance(_tri, _pos) ); }

#endif