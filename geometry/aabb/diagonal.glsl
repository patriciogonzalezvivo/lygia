
#include "aabb.glsl"

/*
contributors: Patrincio Gonzalez Vivo
description: return the diagonal vector of a AABB
use: <float> diagonal(<AABB> box ) 
*/

#ifndef FNC_AABB_DIAGONAL
#define FNC_AABB_DIAGONAL

vec3 diagonal(const in AABB box) { return abs(box.max - box.min); }

#endif