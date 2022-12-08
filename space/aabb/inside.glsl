
#include "../aabb.glsl"

/*
original_author: P
description: Compute if point is inside AABB
use: <bool> AABBinside(<AABB> box, <vec3> point ) 
*/

#ifndef FNC_AABB_INSIDE
#define FNC_AABB_INSIDE

bool AABBinside(const in AABB box, const in vec3 point ) {
    return  all( lessThanEqual(point, box.max) ) && 
            all( lessThan(box.min, point) );
}

#endif