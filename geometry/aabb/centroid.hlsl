#include "aabb.cuh"

/*
original_author: Patricio Gonzalez Vivo
description: return center of a AABB
use: <float3> centrood(<AABB> box) 
*/

#ifndef FNC_AABB_CENTROID
#define FNC_AABB_CENTROID
float3 centroid(AABB _box) { return (_box.min + _box.max) * 0.5; }
#endif