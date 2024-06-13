#include "diagonal.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Square an AABB using the longest side
use: <void> AABBsquare(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AABB_SQUARE
#define FNC_AABB_SQUARE

void square(AABB& _box) {
    float3 diag     = diagonal(_box) * 0.5f;
    float3 cntr     = _box.min + diag;
    float  mmax     = max( abs(diag.x), max( abs(diag.y), abs(diag.z) ) );
    _box.max         = cntr + mmax;
    _box.min         = cntr - mmax;
}

#endif