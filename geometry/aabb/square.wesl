import lygia::geometry::aabb::diagonal::diagonal;
import lygia::geometry::aabb::aabb::AABB;

/*
contributors: Patricio Gonzalez Vivo
description: Square an AABB using the longest side
use: <void> square(<ptr<function, AABB>> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn square(box: ptr<function, AABB>) {
    let diag = diagonal(*box) * 0.5;
    let cntr = (*box).min + diag;
    let mmax = max(abs(diag.x), max(abs(diag.y), abs(diag.z)));
    (*box).max = cntr + mmax;
    (*box).min = cntr - mmax;
}
