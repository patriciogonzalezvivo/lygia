
#include "aabb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Return the diagonal vector of a AABB
use: <float> diagonal(<AABB> box ) 
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AABB_DIAGONAL
#define FNC_AABB_DIAGONAL

vec3 diagonal(const in AABB box) { return abs(box.max - box.min); }

#endif