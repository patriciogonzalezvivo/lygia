#include "aabb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Expand AABB
use: <bool> expand(<AABB> box, <AABB|vec3|float> point )
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn expand(_box: AABB, _value: f32) {
    _box.min -= _value;
    _box.max += _value;
}

fn expanda(_box: AABB, _point: vec3f) {
    _box.min = min(_box.min, _point);
    _box.max = max(_box.max, _point);
}

fn expandb(_box: AABB, _other: AABB) {
    expand(_box, _other.min); 
    expand(_box, _other.max);
}
