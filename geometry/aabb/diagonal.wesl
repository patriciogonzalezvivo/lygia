
import lygia::geometry::aabb::aabb::AABB;

/*
contributors: Patricio Gonzalez Vivo
description: Return the diagonal vector of an AABB
use: <vec3f> diagonal(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn diagonal(box: AABB) -> vec3f {
    return abs(box.max - box.min);
}
