
#include "aabb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Expand AABB
use: <bool> expand(<AABB> box, <AABB|vec3|float> point )
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AABB_EXPAND
#define FNC_AABB_EXPAND

void expand(inout AABB _box, const in float _value ) {
    _box.min -= _value;
    _box.max += _value;
}

void expand(inout AABB _box, const in vec3 _point ) {
    _box.min = min(_box.min, _point);
    _box.max = max(_box.max, _point);
}

void expand(inout AABB _box, const in AABB _other ) {
    expand(_box, _other.min); 
    expand(_box, _other.max);
}

#endif