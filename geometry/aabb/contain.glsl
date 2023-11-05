
#include "aabb.glsl"

/*
contributors: P
description: Compute if point is inside AABB
use: <bool> contain(<AABB> box, <vec3> point ) 
*/

#ifndef FNC_AABB_CONTAIN
#define FNC_AABB_CONTAIN

bool contain(const in AABB box, const in vec3 point ) {
    return  all( lessThanEqual(point, box.max) ) && 
            all( lessThan(box.min, point) );
}

#endif