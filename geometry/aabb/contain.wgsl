#include "aabb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Compute if point is inside AABB
use: <bool> contain(<AABB> box, <vec3> point ) 
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn contain(box: AABB, point: vec3f) -> bool {
    return  all( lessThanEqual(point, box.max) ) && 
            all( lessThan(box.min, point) );
}
