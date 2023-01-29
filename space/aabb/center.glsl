
#include "diagonal.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: return center of a AABB
use: <vec3> AABBcenter(<AABB> box) 
*/

#ifndef FNC_AABB_CENTER
#define FNC_AABB_CENTER

vec3 AABBcenter(const in AABB box) { return diagonal(box) * 0.5; }

#endif