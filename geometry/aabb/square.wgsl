#include "diagonal.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Square an AABB using the longest side
use: <void> square(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn square(_box: AABB&) {
    let diag = diagonal(_box) * 0.5;
    let cntr = _box.min + diag;
    let mmax = max( abs(diag.x), max( abs(diag.y), abs(diag.z) ) );
    _box.max    = cntr + mmax;
    _box.min    = cntr - mmax;
}
