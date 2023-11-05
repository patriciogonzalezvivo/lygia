
#include "aabb.hlsl"

/*
contributors: Patrincio Gonzalez Vivo
description: return the diagonal vector of a AABB
use: <float> diagonal(<AABB> box ) 
*/

#ifndef FNC_AABB_DIAGONAL
#define FNC_AABB_DIAGONAL

float3 diagonal(const AABB box) { return abs(box.max - box.min); }

#endif