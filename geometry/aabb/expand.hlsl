
#include "aabb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Expand AABB
use: <bool> expand(<AABB> box, <AABB|float3|float> point ) 
*/

#ifndef FNC_AABB_EXPAND
#define FNC_AABB_EXPAND

void expand(inout AABB _box, const float _value ) {
    _box.min -= _value;
    _box.max += _value;
}

void expand(inout AABB _box, const float3 _point ) {
    _box.min = min(_box.min, _point);
    _box.max = max(_box.max, _point);
}

void expand(inout AABB _box, const AABB _other ) {
    expand(_box, _other.min); 
    expand(_box, _other.max);
}

#endif