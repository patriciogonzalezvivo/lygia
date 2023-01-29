
#include "../aabb.glsl"

/*
original_author: Patrincio Gonzalez Vivo
description: return the diagonal vector of a AABB
use: <float> AABBdiagonal(<AABB> box ) 
*/

#ifndef FNC_AABB_DIAGONAL
#define FNC_AABB_DIAGONAL

vec3 AABBdiagonal(const in AABB box) { return abs(box.max - box.min); }

#endif