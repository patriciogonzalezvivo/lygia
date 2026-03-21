#include "aabb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Return center of a AABB
use: <vec3> centroid(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn centroid(_box: AABB) -> vec3f { return (_box.min + _box.max) * 0.5; }
