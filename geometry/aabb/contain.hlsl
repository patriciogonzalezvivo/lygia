
#include "aabb.hlsl"

/*
contributors: P
description: Compute if point is inside AABB
use: <bool> inside(<AABB> box, <float3> point ) 
*/

#ifndef FNC_AABB_CONTAIN
#define FNC_AABB_CONTAIN

bool contain(const in AABB _box, const in float3 _point ) {
    return  (_point.x >= _box.min.x && _point.x <= _box.max.x) &&
            (_point.y >= _box.min.y && _point.y <= _box.max.y) &&
            (_point.z >= _box.min.z && _point.z <= _box.max.z);
}

#endif