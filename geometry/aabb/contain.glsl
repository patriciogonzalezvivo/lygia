
#include "aabb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Compute if point is inside AABB
use: <bool> contain(<AABB> box, <vec3> point ) 
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AABB_CONTAIN
#define FNC_AABB_CONTAIN

bool contain(const in AABB box, const in vec3 point ) {
    return  all( lessThanEqual(point, box.max) ) && 
            all( lessThan(box.min, point) );
}

#endif