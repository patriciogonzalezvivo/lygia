#include "aabb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: return center of a AABB
use: <vec3> centroid(<AABB> box) 
*/

#ifndef FNC_AABB_CENTROID
#define FNC_AABB_CENTROID
vec3 centroid(const in AABB _box) { return (_box.min + _box.max) * 0.5; }
#endif