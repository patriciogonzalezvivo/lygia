
#include "../aabb.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Expand AABB
use: <bool> AABBexpand(<AABB> box, <AABB|float3|float> point ) 
*/

#ifndef FNC_AABB_EXPAND
#define FNC_AABB_EXPAND

void AABBexpand(inout AABB _box, const float _value ) {
    _box.min -= _value;
    _box.max += _value;
}

void AABBexpand(inout AABB _box, const float3 _point ) {
    _box.min.x = min(_box.min.x, _point.x);
    _box.max.x = max(_box.max.x, _point.x);
    _box.min.y = min(_box.min.y, _point.y);
    _box.max.y = max(_box.max.y, _point.y);
    _box.min.z = min(_box.min.z, _point.z);
    _box.max.z = max(_box.max.z, _point.z);
}

void AABBexpand(inout AABB _box, const AABB _other ) {
    AABBexpand(_box, _other.min); 
    AABBexpand(_box, _other.max);
}

#endif