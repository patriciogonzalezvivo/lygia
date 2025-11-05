
import lygia::geometry::aabb::aabb::AABB;

/*
contributors: Patricio Gonzalez Vivo
description: Expand AABB
use: <void> expand(<ptr<function, AABB>> box, <AABB|vec3f|f32> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn expand(box: ptr<function, AABB>, value: f32) {
    (*box).min -= value;
    (*box).max += value;
}

fn expand2(box: ptr<function, AABB>, point: vec3f) {
    (*box).min = min((*box).min, point);
    (*box).max = max((*box).max, point);
}

fn expand3(box: ptr<function, AABB>, other: AABB) {
    expand2(box, other.min);
    expand2(box, other.max);
}
