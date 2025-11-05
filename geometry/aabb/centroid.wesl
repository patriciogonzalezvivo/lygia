
import lygia::geometry::aabb::aabb::AABB;

/*
contributors: Patricio Gonzalez Vivo
description: Return the centroid of an AABB
use: <vec3f> centroid(<AABB> box)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn centroid(box: AABB) -> vec3f {
    return (box.min + box.max) * 0.5;
}
