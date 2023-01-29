
#include "../aabb.glsl"

/*
original_author: P
description: Compute if point is inside AABB
use: <bool> AABBcontain(<AABB> box, <vec3> point ) 
*/

#ifndef FNC_AABB_CONTAIN
#define FNC_AABB_CONTAIN

bool AABBcontain(const in AABB box, const in vec3 point ) {
    return  all( lessThanEqual(point, box.max) ) && 
            all( lessThan(box.min, point) );
}

#endif