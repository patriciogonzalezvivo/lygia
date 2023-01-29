
#include "diagonal.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: return center of a AABB
use: <float3> AABBcenter(<AABB> box) 
*/

#ifndef FNC_AABB_CENTER
#define FNC_AABB_CENTER

float3 AABBcenter(const in AABB box) { return diagonal(box) * 0.5; }

#endif